{code}
"""
Josephson Junction Solver - 合并优化版
基于Floquet理论和非平衡格林函数方法
支持：CPR, DC_IV, ABS, SPECTRA 计算
特性：稀疏矩阵、增量哈密顿量、多级缓存、自适应边带、无序区域选择
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import diags, block_diag, lil_matrix, csc_matrix, identity, issparse, eye as sparse_eye
from scipy.sparse.linalg import inv as sparse_inv, spsolve, factorized
import scipy.sparse as sp
from joblib import Parallel, delayed
import pandas as pd
import os
import time
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt
import hashlib
import functools
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==================== 1. HamiltonianBuilder ====================

class HamiltonianBuilder:
    """
    哈密顿量构建器
    - 支持稀疏/稠密矩阵自动选择
    - 支持增量Floquet哈密顿量扩建
    - 支持多种无序类型和区域选择
    """
    
    def __init__(self, params):
        # Validate disorder type
        if 'disorder_type' in params:
            valid_disorders = ['gaussian', 'smooth', 'random_typeI', 'random_typeII', 
                              'nonhermitian', 'none', 'from_file']
            if params['disorder_type'] not in valid_disorders:
                raise ValueError(f"Invalid disorder_type: {params['disorder_type']}")
        
        # Validate disorder region (from Part 2)
        if 'disorder_region' in params:
            valid_regions = ['all', 'left_lead', 'junction', 'right_lead', 'all_leads']
            if params['disorder_region'] not in valid_regions:
                raise ValueError(f"Invalid disorder_region: {params['disorder_region']}")
        
        # Physical parameters
        self.t = params['t']
        self.delta = params['delta']
        self.mu = params['mu']
        self.mu_lead = params['mu_lead']
        self.B = params['B']
        self.alpha = params['alpha']
        
        # Junction parameters
        self.N_SC = params['N_SC']
        self.N_junction = params['N_junction']
        self.v_tau = params['v_tau']
        
        # Disorder parameters
        self.disorder_type = params.get('disorder_type', 'none')
        self.disorder_region = params.get('disorder_region', 'all')  # From Part 2
        self.disorder_strength = params.get('disorder_strength', 1.0)
        
        # Gaussian disorder
        self.V_gau = params.get('Vdis_gau', 0.0)
        self.X_gau = params.get('Xdis_gau', 50)
        self.decayL_gau = params.get('decayL_gau', 50)
        
        # Smooth disorder
        self.decayL_smooth = params.get('decayL_smooth', 50)
        self.Vmax_smooth = params.get('Vdis_smooth', 0.3)
        self.Vd_smooth = params.get('Vd_smooth', 0.0)
        
        # Random type I
        self.Num_Rd1 = params.get('N_imp1', 52)
        self.lambda_Rd1 = params.get('lambda_imp1', 18.0)
        self.V0_Rd1 = params.get('V0_imp1', 0.0)
        
        # Random type II
        self.Nd_Rd2 = params.get('Nd_imp2', 10.0)
        self.lambda_Rd2 = params.get('lambda_imp2', 20.0)
        self.V0_Rd2 = params.get('V0_imp2', 0.0)
        self.a0 = params.get('a0', 10.0)
        
        # Non-hermitian
        self.nonH_eta = params.get('nonH_eta', 1.5e-3)
        
        # From file
        self.disorder_file = params.get('disorder_file', 'disorder_data.txt')
        
        # Floquet parameters
        self.max_sidebands = params['max_sidebands']
        
        # Initialize Pauli matrices
        self._init_pauli_matrices()
        
        # Calculate total system length
        self.total_sites = 2 * self.N_SC + self.N_junction
        self.select_mid_i = params.get('mid_site_i', self.N_SC + self.N_junction // 2)
        
        # Initialize disorder distribution (with region support)
        self.disorder_distribution = self._calculate_disorder_distribution()
        
        # Sparse matrix configuration (from Part 1)
        self.use_sparse = params.get('use_sparse', True)
        self.sparse_threshold = params.get('sparse_threshold', 0.3)
        self.sparse_depth_limit = params.get('sparse_depth_limit', 5)
        
        # Incremental expansion cache (from Part 1)
        self.prev_sidebands = None
        self.cached_g0_inv_FKs = None
        self.cached_hop_FKs = None
        self.incremental_cache = {}
        
        logging.info(f"HamiltonianBuilder initialized, disorder: {self.disorder_type}, "
                    f"region: {self.disorder_region}, sparse: {self.use_sparse}")
    
    def _init_pauli_matrices(self):
        """Initialize Pauli matrices for spin and Nambu space"""
        # Spin space
        self.sigma_0 = np.eye(2)
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])
        
        # Nambu space
        self.tau_0 = np.eye(2)
        self.tau_x = np.array([[0, 1], [1, 0]])
        self.tau_y = np.array([[0, -1j], [1j, 0]])
        self.tau_z = np.array([[1, 0], [0, -1]])
        self.tau_plus = (self.tau_x + 1j * self.tau_y) / 2
        self.tau_minus = (self.tau_x - 1j * self.tau_y) / 2
        self.tau_e = np.array([[1, 0], [0, 0]])  # Electron block
        self.tau_h = np.array([[0, 0], [0, 1]])  # Hole block
    
    def _get_disorder_sites(self):
        """Determine which sites to apply disorder (from Part 2)"""
        if self.disorder_region == 'all':
            return range(self.total_sites)
        elif self.disorder_region == 'left_lead':
            return range(0, self.N_SC)
        elif self.disorder_region == 'junction':
            return range(self.N_SC, self.N_SC + self.N_junction)
        elif self.disorder_region == 'right_lead':
            return range(self.N_SC + self.N_junction, self.total_sites)
        elif self.disorder_region == 'all_leads':
            return [*range(0, self.N_SC), *range(self.N_SC + self.N_junction, self.total_sites)]
        else:
            return []
    
    def _calculate_disorder_distribution(self):
        """Calculate disorder distribution for selected region"""
        sites = self._get_disorder_sites()
        disorder_array = np.zeros(self.total_sites)
        
        if self.disorder_type == 'smooth':
            for i in sites:
                if i < self.N_SC:
                    disorder_array[i] = self.Vmax_smooth * (
                        1 - self.Vd_smooth * np.exp(i / self.decayL_smooth))
                elif i >= self.N_junction + self.N_SC:
                    disorder_array[i] = self.Vmax_smooth * (
                        1 - self.Vd_smooth * np.exp(-(i - self.N_junction) / self.decayL_smooth))
                else:  # inside junction
                    disorder_array[i] = self.Vmax_smooth * (1 - self.Vd_smooth)
        
        elif self.disorder_type == 'gaussian':
            for i in sites:
                disorder_array[i] = self.V_gau * np.exp(-(i - self.X_gau)**2 / self.decayL_gau**2)
        
        elif self.disorder_type == 'nonhermitian':
            for i in sites:
                disorder_array[i] = -1j * self.nonH_eta
        
        elif self.disorder_type == 'random_typeI':
            disorder_array = self._generate_random_typeI_disorder(sites)
        
        elif self.disorder_type == 'random_typeII':
            disorder_array = self._generate_random_typeII_disorder(sites)
        
        elif self.disorder_type == 'from_file':
            disorder_array = self._load_disorder_from_file()
        
        return disorder_array
    
    def _load_disorder_from_file(self):
        """Load disorder distribution from file"""
        try:
            if self.disorder_file.endswith('.csv'):
                df = pd.read_csv(self.disorder_file)
                if 'disorder_value' in df.columns:
                    disorder_array = df['disorder_value'].values
                else:
                    disorder_array = df.iloc[:, 0].values
            elif self.disorder_file.endswith('.txt') or self.disorder_file.endswith('.dat'):
                disorder_array = np.loadtxt(self.disorder_file)
            elif self.disorder_file.endswith('.npy'):
                disorder_array = np.load(self.disorder_file)
            else:
                disorder_array = np.loadtxt(self.disorder_file)
            
            # Check length match
            if len(disorder_array) != self.total_sites:
                logging.warning(f"Disorder length mismatch: {len(disorder_array)} vs {self.total_sites}")
                if len(disorder_array) > self.total_sites:
                    disorder_array = disorder_array[:self.total_sites]
                else:
                    disorder_array = np.pad(disorder_array, 
                                          (0, self.total_sites - len(disorder_array)), 'constant')
            
            return disorder_array
            
        except Exception as e:
            logging.error(f"Failed to load disorder: {str(e)}")
            return np.zeros(self.total_sites)
    
    def _generate_random_typeI_disorder(self, sites):
        """Generate charge impurity disorder (Part 2 style)"""
        disorder_array = np.zeros(self.total_sites)
        rng = np.random.default_rng()
        
        impurity_positions = rng.choice(list(sites), self.Num_Rd1)
        impurity_signs = np.array([(-1)**n for n in range(self.Num_Rd1)])
        
        S = np.zeros(self.total_sites)
        for i in sites:
            for j in range(self.Num_Rd1):
                distance = abs(i - impurity_positions[j])
                S[i] += impurity_signs[j] * np.exp(-distance / self.lambda_Rd1)
        
        # Normalize
        S_avg = np.mean(S)
        S_centered = S - S_avg
        variance = np.mean(S_centered**2)
        N_V = np.sqrt(variance) / self.V0_Rd1 if variance > 0 else 1.0
        
        for i in sites:
            disorder_array[i] = (self.V0_Rd1 / N_V) * S_centered[i]
        
        return disorder_array
    
    def _generate_random_typeII_disorder(self, sites):
        """Generate random amplitude disorder (Part 2 style)"""
        disorder_array = np.zeros(self.total_sites)
        
        region_length = len(sites) * self.a0 / 1000  # microns
        N_d = int(self.Nd_Rd2 * region_length)
        
        rng = np.random.default_rng()
        
        min_pos = min(sites) * self.a0 if sites else 0
        max_pos = max(sites) * self.a0 if sites else 0
        impurity_positions = rng.uniform(min_pos, max_pos, N_d)
        impurity_amplitudes = rng.normal(0, 1, N_d)
        
        for i in sites:
            x_i = i * self.a0
            potential_sum = 0.0
            for j in range(N_d):
                distance = abs(x_i - impurity_positions[j])
                potential_sum += impurity_amplitudes[j] * np.exp(-distance / self.lambda_Rd2)
            disorder_array[i] = self.V0_Rd2 * potential_sum
        
        return disorder_array
    
    def disorder_potential(self, site_index):
        """
        Return disorder potential matrix at given site
        disorder_strength applied HERE (统一在内部处理)
        """
        if 0 <= site_index < len(self.disorder_distribution):
            V_s = self.disorder_distribution[site_index] * self.disorder_strength
            
            if self.disorder_type == 'nonhermitian':
                return V_s * np.kron(self.tau_0, self.sigma_0)
            else:
                return V_s * np.kron(self.tau_z, self.sigma_0)
        else:
            return np.zeros((4, 4))
    
    def block_hamiltonian(self, phi, region_type):
        """Construct Hamiltonian block for given region type"""
        # Note: Zeeman term formula kept as in Part 1 for compatibility
        # TODO: Verify this formula with user later
        onsite = 2 * self.t - self.mu
        H00 = onsite * np.kron(self.tau_z, self.sigma_0) + \
              self.B * np.kron(self.tau_z, self.sigma_x) * np.sqrt(self.mu_lead**2 + self.delta**2)
        H01 = -self.t * np.kron(self.tau_z, self.sigma_0) + \
              1j * self.alpha * np.kron(self.tau_z, self.sigma_y)
        
        if region_type == 'SC':
            H00 = H00 + self.delta * (
                np.exp(1j * phi) * np.kron(self.tau_plus, 1j*self.sigma_y) +
                np.exp(-1j * phi) * np.kron(self.tau_minus, -1j*self.sigma_y))
        
        return H00, H01
    
    def construct_hamiltonian(self, phi):
        """Construct full Hamiltonian for SNS junction"""
        H00_L, H01_L = self.block_hamiltonian(0, 'SC')
        H00_R, H01_R = self.block_hamiltonian(phi, 'SC')
        H00_J, H01_J = self.block_hamiltonian(0, 'NM')
        
        return {
            'H00_L': H00_L, 'H01_L': H01_L,
            'H00_R': H00_R, 'H01_R': H01_R,
            'H00_J': H00_J, 'H01_J': H01_J
        }
    
    def build_slice_ham(self, Vbias, ham, max_sidebands):
        """
        Build Floquet Hamiltonian slices with incremental expansion support
        """
        # Check if incremental expansion is possible
        can_increment = (
            self.prev_sidebands is not None and 
            max_sidebands > self.prev_sidebands and
            self.cached_g0_inv_FKs is not None and 
            self.cached_hop_FKs is not None
        )
        
        if can_increment:
            logging.info(f"Using incremental expansion: {self.prev_sidebands} -> {max_sidebands}")
            g0_inv_FKs, hop_FKs = self._incremental_expand_slice_ham(Vbias, ham, max_sidebands)
        else:
            logging.info(f"Full build mode: {max_sidebands} sidebands")
            g0_inv_FKs, hop_FKs = self._full_build_slice_ham(Vbias, ham, max_sidebands)
        
        # Update cache
        self.prev_sidebands = max_sidebands
        self.cached_g0_inv_FKs = g0_inv_FKs
        self.cached_hop_FKs = hop_FKs
        
        return g0_inv_FKs, hop_FKs
    
    def _incremental_expand_slice_ham(self, Vbias, ham, new_max_sidebands):
        """Incremental expansion of Floquet Hamiltonian (Part 1 optimized)"""
        old_max_sidebands = self.prev_sidebands
        old_sideN = 2 * old_max_sidebands + 1
        new_sideN = 2 * new_max_sidebands + 1
        dim_old = 4 * old_sideN
        dim_new = 4 * new_sideN
        
        old_g0_inv_FKs = self.cached_g0_inv_FKs
        old_hop_FKs = self.cached_hop_FKs
        
        g0_inv_FKs = []
        hop_FKs = []
        
        SNS_Len = self.N_SC * 2 + self.N_junction
        select_mid_i = self.select_mid_i
        
        for site_i in range(SNS_Len):
            old_g0_inv = old_g0_inv_FKs[site_i]
            old_hop = old_hop_FKs[site_i]
            
            # Determine region
            if self.N_SC <= site_i < self.N_SC + self.N_junction:
                Ham_onsite = ham['H00_J']
                Ham_hop = ham['H01_J']
                Hop_e = Ham_hop * np.kron(self.tau_e, np.ones((2, 2))) * self.v_tau
                Hop_h = -1 * Ham_hop * np.kron(self.tau_h, np.ones((2, 2))) * self.v_tau
                is_junction = True
            elif site_i < self.N_SC:
                Ham_onsite = ham['H00_L']
                Ham_hop = ham['H01_L']
                is_junction = False
            else:
                Ham_onsite = ham['H00_R']
                Ham_hop = ham['H01_R']
                is_junction = False
            
            # Add disorder (strength applied inside disorder_potential)
            Ham_onsite = Ham_onsite + self.disorder_potential(site_i)
            
            # Expand g0_inv_FK
            num_bands_to_add = new_max_sidebands - old_max_sidebands
            diag_blocks = []
            
            # Upper new blocks
            for band_i in range(-new_max_sidebands, -old_max_sidebands):
                diag_block = Ham_onsite - band_i * Vbias * np.eye(4)
                diag_blocks.append(diag_block)
            
            # Middle old blocks
            if sp.issparse(old_g0_inv):
                for i in range(old_sideN):
                    start = i * 4
                    end = (i + 1) * 4
                    diag_blocks.append(old_g0_inv[start:end, start:end].toarray())
            else:
                for n in range(old_sideN):
                    idx = slice(n*4, (n+1)*4)
                    diag_blocks.append(old_g0_inv[idx, idx])
            
            # Lower new blocks
            for band_i in range(old_max_sidebands+1, new_max_sidebands+1):
                diag_block = Ham_onsite - band_i * Vbias * np.eye(4)
                diag_blocks.append(diag_block)
            
            # Build new block diagonal matrix
            if self.use_sparse and dim_new > 20:
                g0_inv_new = block_diag(diag_blocks, format='csc')
            else:
                g0_inv_new = np.zeros((dim_new, dim_new), dtype=np.complex128)
                for i, block in enumerate(diag_blocks):
                    start_idx = i * 4
                    end_idx = (i+1) * 4
                    g0_inv_new[start_idx:end_idx, start_idx:end_idx] = block
            
            # Expand hop_FK
            if self.use_sparse and dim_new > 20:
                hop_new = lil_matrix((dim_new, dim_new), dtype=np.complex128)
            else:
                hop_new = np.zeros((dim_new, dim_new), dtype=np.complex128)
            
            # Place old hopping matrix in center
            old_start = num_bands_to_add * 4
            old_end = old_start + dim_old
            if sp.issparse(hop_new):
                hop_new[old_start:old_end, old_start:old_end] = old_hop
            else:
                hop_new[old_start:old_end, old_start:old_end] = old_hop
            
            # Add coupling for special sites
            is_special_site = (site_i == select_mid_i)
            if is_special_site and is_junction:
                for n in range(new_sideN):
                    band_i = n - new_max_sidebands
                    start_idx = n * 4
                    end_idx = (n+1)*4
                    
                    if band_i != new_max_sidebands and Vbias >= 0:
                        next_idx = (n+1) * 4
                        if next_idx < dim_new:
                            if sp.issparse(hop_new):
                                hop_new[start_idx:end_idx, next_idx:next_idx+4] = Hop_e
                            else:
                                hop_new[start_idx:end_idx, next_idx:next_idx+4] = Hop_e
                    
                    if band_i != -new_max_sidebands and Vbias >= 0:
                        prev_idx = (n-1) * 4
                        if prev_idx >= 0:
                            if sp.issparse(hop_new):
                                hop_new[start_idx:end_idx, prev_idx:prev_idx+4] = Hop_h.conj().T
                            else:
                                hop_new[start_idx:end_idx, prev_idx:prev_idx+4] = Hop_h.conj().T
            
            if sp.issparse(hop_new):
                hop_new = hop_new.tocsc()
            
            g0_inv_FKs.append(g0_inv_new)
            hop_FKs.append(hop_new)
        
        return g0_inv_FKs, hop_FKs
    
    def _full_build_slice_ham(self, Vbias, ham, max_sidebands):
        """Full build of Floquet Hamiltonian (Part 1 optimized with Part 2 disorder)"""
        sideN = 2 * max_sidebands + 1
        dim = 4 * sideN
        g0_inv_FKs = []
        hop_FKs = []
        
        SNS_Len = self.N_SC * 2 + self.N_junction
        SN_Len = self.N_SC + self.N_junction
        
        select_mid_i = self.select_mid_i
        
        for site_i in range(SNS_Len):
            # Determine region
            if self.N_SC <= site_i < SN_Len:
                Ham_onsite = ham['H00_J']
                Ham_hop = ham['H01_J']
                Hop_e = Ham_hop * np.kron(self.tau_e, np.ones((2, 2))) * self.v_tau
                Hop_h = -1 * Ham_hop * np.kron(self.tau_h, np.ones((2, 2))) * self.v_tau
                is_junction = True
            elif site_i < self.N_SC:
                Ham_onsite = ham['H00_L']
                Ham_hop = ham['H01_L']
                is_junction = False
            else:
                Ham_onsite = ham['H00_R']
                Ham_hop = ham['H01_R']
                is_junction = False
            
            # Add disorder (strength applied inside disorder_potential)
            Ham_onsite = Ham_onsite + self.disorder_potential(site_i)
            
            # Build matrices
            if self.use_sparse and dim > 20:
                # Sparse path
                diag_blocks = []
                for n in range(sideN):
                    band_i = n - max_sidebands
                    diag_block = Ham_onsite - band_i * Vbias * np.eye(4)
                    diag_blocks.append(diag_block)
                
                g0_inv_FK0 = block_diag(diag_blocks, format='csc')
                hop_FK0 = lil_matrix((dim, dim), dtype=np.complex128)
                
                for n in range(sideN):
                    band_i = n - max_sidebands
                    idx = slice(n * 4, (n + 1) * 4)
                    
                    is_special_site = (site_i == select_mid_i)
                    
                    if is_special_site and is_junction:
                        if band_i != max_sidebands and Vbias >= 0:
                            next_idx = slice((n + 1) * 4, (n + 2) * 4)
                            hop_FK0[idx, next_idx] = Hop_e
                        if band_i != -max_sidebands and Vbias >= 0:
                            prev_idx = slice((n - 1) * 4, n * 4)
                            hop_FK0[idx, prev_idx] = Hop_h.conj().T
                    else:
                        hop_FK0[idx, idx] = Ham_hop
                
                hop_FK0 = hop_FK0.tocsc()
            else:
                # Dense path
                g0_inv_FK0 = np.zeros((dim, dim), dtype=np.complex128)
                hop_FK0 = np.zeros_like(g0_inv_FK0)
                
                for n in range(sideN):
                    band_i = n - max_sidebands
                    idx = slice(n * 4, (n + 1) * 4)
                    g0_inv_FK0[idx, idx] = Ham_onsite - band_i * Vbias * np.eye(4)
                    
                    is_special_site = (site_i == select_mid_i)
                    
                    if is_special_site and is_junction:
                        if band_i != max_sidebands and Vbias >= 0:
                            next_idx = slice((n + 1) * 4, (n + 2) * 4)
                            hop_FK0[idx, next_idx] = Hop_e
                        if band_i != -max_sidebands and Vbias >= 0:
                            prev_idx = slice((n - 1) * 4, n * 4)
                            hop_FK0[idx, prev_idx] = Hop_h.conj().T
                    else:
                        hop_FK0[idx, idx] = Ham_hop
            
            g0_inv_FKs.append(g0_inv_FK0)
            hop_FKs.append(hop_FK0)
        
        return g0_inv_FKs, hop_FKs


# ==================== 2. GreenFunctionCalculator ====================

class GreenFunctionCalculator:
    """
    格林函数计算器
    - 兼容稀疏和稠密矩阵
    - 带矩阵分解缓存
    - 修复了Part 2的变量定义Bug
    """
    
    def __init__(self, params):
        self.recursion_depth = params['recursion_depth']
        self.eta = params['eta']
        self.mu_lead = params['mu_lead']
        self.delta = params['delta']
        self.unit_factor_nA = params.get('unit_factor_nA', 
                                        2 * 1.6e-19 * 1e-3 / (4.13e-15) * 1e9)
        
        # Sparse configuration
        self.use_sparse = params.get('use_sparse', True)
        self.sparse_threshold = params.get('sparse_threshold', 0.3)
        self.sparse_depth_limit = params.get('sparse_depth_limit', 5)
        
        # Factorization cache
        self.factorized_cache = {}
        
        logging.info(f"GreenFunctionCalculator initialized, sparse: {self.use_sparse}")
    
    def fermi_dirac(self, omega):
        """Fermi-Dirac distribution"""
        kBT = 5e-3 * self.delta
        return 1.0 / (1.0 + np.exp((omega - self.mu_lead) / kBT))
    
    def surface_gf_sc(self, omega, H00, H01, direction='left'):
        """Surface Green's function for superconducting lead"""
        E0 = omega + 1j * self.eta
        dim = H00.shape[0]
        eps = eps_s = H00.copy()
        
        if direction == 'right':
            alpha = H01
            beta = H01.conj().T
        else:
            alpha = H01.conj().T
            beta = H01
        
        # Recursive calculation
        for _ in range(self.recursion_depth):
            g = la.inv(E0 * np.eye(dim) - eps)
            eps_s = eps_s + alpha @ g @ beta
            eps = eps + alpha @ g @ beta + beta @ g @ alpha
            
            alpha_new = alpha @ g @ alpha
            beta_new = beta @ g @ beta
            
            if np.linalg.norm(alpha_new) < 1e-12:
                break
            alpha, beta = alpha_new, beta_new
        
        g_surface = la.inv(E0 * np.eye(dim) - eps_s)
        hgh = H01 @ g_surface @ H01.conj().T if direction == 'right' else \
              H01.conj().T @ g_surface @ H01
        
        return g_surface, hgh
    
    def recursive_sweep(self, omega, sE_initial, range_sets, hop_FKs, g0_inv_FKs):
        """
        Recursive sweep for self-energy calculation
        FIXED: Properly initialize GL_i, GR_i variables (Part 1 fix)
        """
        range_1, range_2, offset_L = range_sets
        sE_L0, sE_R0, sE_Lf0, sE_Rf0 = sE_initial
        
        # Get matrix dimension
        first_mat = g0_inv_FKs[0]
        dim = first_mat.shape[0]
        
        # Create identity matrix (sparse/dense compatible)
        if sp.issparse(g0_inv_FKs[0]):
            E0_I0 = (omega + 1j * self.eta) * sp.eye(dim, format='csc', dtype=np.complex128)
        else:
            E0_I0 = (omega + 1j * self.eta) * np.eye(dim, dtype=np.complex128)
        
        # Initialize lists
        g0s_invL = []
        sELs = []
        sERs = []
        sELs_less = []
        sERs_less = []
        
        # CRITICAL FIX: Initialize variables before loop (from Part 1)
        GL_i = None
        GR_i = None
        GL_less_i = None
        GR_less_i = None
        
        for recur_i in range(range_1, range_2):
            recur_i_R = range_2 - 1 - recur_i + offset_L
            
            # Get current matrices
            g0_inv_i = g0_inv_FKs[recur_i]
            g0_inv_iR = g0_inv_FKs[recur_i_R]
            alpha_i = hop_FKs[recur_i]
            alpha_iR = hop_FKs[recur_i_R]
            
            # Initial self-energy handling
            if recur_i == range_1 or recur_i_R == range_2 - 1:
                sEL_i = sE_L0
                sER_i = sE_R0
                sEL_less_i = sE_Lf0.conj().T - sE_Lf0
                sER_less_i = sE_Rf0.conj().T - sE_Rf0
            else:
                # Use sparse matrix multiplication when possible
                if sp.issparse(alpha_i) and sp.issparse(GL_i):
                    sEL_i = alpha_i.conj().T @ GL_i @ alpha_i
                else:
                    if sp.issparse(GL_i):
                        GL_i = GL_i.toarray()
                    if sp.issparse(alpha_i):
                        alpha_i = alpha_i.toarray()
                    sEL_i = alpha_i.conj().T @ GL_i @ alpha_i
                
                if sp.issparse(alpha_iR) and sp.issparse(GR_i):
                    sER_i = alpha_iR @ GR_i @ alpha_iR.conj().T
                else:
                    if sp.issparse(GR_i):
                        GR_i = GR_i.toarray()
                    if sp.issparse(alpha_iR):
                        alpha_iR = alpha_iR.toarray()
                    sER_i = alpha_iR @ GR_i @ alpha_iR.conj().T
                
                # Lesser components
                if sp.issparse(alpha_i) and sp.issparse(GL_less_i):
                    sEL_less_i = alpha_i.conj().T @ GL_less_i @ alpha_i
                else:
                    if sp.issparse(GL_less_i):
                        GL_less_i = GL_less_i.toarray()
                    if sp.issparse(alpha_i):
                        alpha_i = alpha_i.toarray()
                    sEL_less_i = alpha_i.conj().T @ GL_less_i @ alpha_i
                
                if sp.issparse(alpha_iR) and sp.issparse(GR_less_i):
                    sER_less_i = alpha_iR @ GR_less_i @ alpha_iR.conj().T
                else:
                    if sp.issparse(GR_less_i):
                        GR_less_i = GR_less_i.toarray()
                    if sp.issparse(alpha_iR):
                        alpha_iR = alpha_iR.toarray()
                    sER_less_i = alpha_iR @ GR_less_i @ alpha_iR.conj().T
            
            # Build inverse Green's functions
            gi_inv_L = E0_I0 - g0_inv_i
            gi_inv_R = E0_I0 - g0_inv_iR
            
            # Matrix inversion with caching
            cache_key_L = (id(gi_inv_L), id(sEL_i))
            if cache_key_L in self.factorized_cache:
                solve_L = self.factorized_cache[cache_key_L]
                GL_i = solve_L(sp.eye(dim, format='csc', dtype=np.complex128))
            else:
                mat_to_inv = gi_inv_L - sEL_i
                if sp.issparse(mat_to_inv):
                    solve_L = factorized(mat_to_inv.tocsc())
                    GL_i = solve_L(sp.eye(dim, format='csc', dtype=np.complex128))
                    self.factorized_cache[cache_key_L] = solve_L
                else:
                    GL_i = la.inv(mat_to_inv)
            
            cache_key_R = (id(gi_inv_R), id(sER_i))
            if cache_key_R in self.factorized_cache:
                solve_R = self.factorized_cache[cache_key_R]
                GR_i = solve_R(sp.eye(dim, format='csc', dtype=np.complex128))
            else:
                mat_to_inv = gi_inv_R - sER_i
                if sp.issparse(mat_to_inv):
                    solve_R = factorized(mat_to_inv.tocsc())
                    GR_i = solve_R(sp.eye(dim, format='csc', dtype=np.complex128))
                    self.factorized_cache[cache_key_R] = solve_R
                else:
                    GR_i = la.inv(mat_to_inv)
            
            # Calculate lesser components
            if sp.issparse(GL_i):
                GL_less_i = GL_i @ sEL_less_i @ GL_i.conj().T
            else:
                GL_less_i = GL_i @ sEL_less_i @ GL_i.conj().T
            
            if sp.issparse(GR_i):
                GR_less_i = GR_i @ sER_less_i @ GR_i.conj().T
            else:
                GR_less_i = GR_i @ sER_less_i @ GR_i.conj().T
            
            # Store results
            sELs.append(sEL_i)
            sERs.append(sER_i)
            sELs_less.append(sEL_less_i)
            sERs_less.append(sER_less_i)
            g0s_invL.append(gi_inv_L)
        
        return g0s_invL, [sELs, sERs, sELs_less, sERs_less]
    
    def compute_self_energies(self, omega, lead_type, ham, max_sidebands, Vbias, 
                             N_SC, N_junction, hop_FKs, g0_inv_FKs):
        """Compute self-energies (compatible with sparse/dense)"""
        sideN = 2 * max_sidebands + 1
        dim = 4 * sideN
        
        # Create dense matrices for self-energies
        sEL = np.zeros((dim, dim), dtype=np.complex128)
        sER = np.zeros_like(sEL)
        sEL_f0 = np.zeros_like(sEL)
        sER_f0 = np.zeros_like(sEL)
        
        for n in range(sideN):
            band_i = n - max_sidebands
            omega_shifted = omega + band_i * Vbias
            f0 = self.fermi_dirac(omega_shifted)
            a = slice(n * 4, n * 4 + 4)
            
            # Left SC lead
            _, sEL_block = self.surface_gf_sc(omega_shifted, ham['H00_L'], 
                                              ham['H01_L'], direction='left')
            sEL[a, a] = sEL_block
            sEL_f0[a, a] = sEL_block * f0
            
            # Right SC lead
            _, sER_block = self.surface_gf_sc(omega_shifted, ham['H00_R'], 
                                              ham['H01_R'], direction='right')
            sER[a, a] = sER_block
            sER_f0[a, a] = sER_block * f0
        
        if lead_type == 'infinite':
            return [None, None], [sEL, sER, sEL_f0, sER_f0]
        
        # Finite lead processing
        sE_initial = [sEL, sER, sEL_f0, sER_f0]
        
        # Process left lead region
        g0_inv_Ld, sE_Ld = self.obtain_recursive_list(
            omega, 'left_lead', sE_initial, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        # Process right lead region
        g0_inv_Rd, sE_Rd = self.obtain_recursive_list(
            omega, 'right_lead', sE_initial, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        # Convert to dense if sparse
        g0_inv_L = g0_inv_Ld[-1].toarray() if sp.issparse(g0_inv_Ld[-1]) else g0_inv_Ld[-1]
        g0_inv_R = g0_inv_Rd[-1].toarray() if sp.issparse(g0_inv_Rd[-1]) else g0_inv_Rd[-1]
        
        sEL_left = sE_Ld[0][-1].toarray() if sp.issparse(sE_Ld[0][-1]) else sE_Ld[0][-1]
        sER_left = sE_Ld[1][0].toarray() if sp.issparse(sE_Ld[1][0]) else sE_Ld[1][0]
        sEL_right = sE_Rd[0][0].toarray() if sp.issparse(sE_Rd[0][0]) else sE_Rd[0][0]
        sER_right = sE_Rd[1][-1].toarray() if sp.issparse(sE_Rd[1][-1]) else sE_Rd[1][-1]
        
        # Calculate Green's functions
        g_retard_L = la.inv(g0_inv_L - sEL_left - sER_left)
        g_retard_R = la.inv(g0_inv_R - sEL_right - sER_right)
        
        # Update self-energies
        sEL = np.zeros_like(sEL)
        sER = np.zeros_like(sER)
        sEL_f0 = np.zeros_like(sEL_f0)
        sER_f0 = np.zeros_like(sER_f0)
        
        hop_LC = hop_FKs[N_SC - 1]
        hop_CR = hop_FKs[N_SC + N_junction]
        
        if sp.issparse(hop_LC):
            hop_LC = hop_LC.toarray()
        if sp.issparse(hop_CR):
            hop_CR = hop_CR.toarray()
        
        for n in range(sideN):
            band_i = n - max_sidebands
            omega_shifted = omega + band_i * Vbias
            f0 = self.fermi_dirac(omega_shifted)
            a = slice(n * 4, n * 4 + 4)
            
            sEL[a, a] = hop_LC[a, a].conj().T @ g_retard_L[a, a] @ hop_LC[a, a]
            sER[a, a] = hop_CR[a, a] @ g_retard_R[a, a] @ hop_CR[a, a].conj().T
            sEL_f0[a, a] = sEL[a, a] * f0
            sER_f0[a, a] = sER[a, a] * f0
        
        return [g_retard_L, g_retard_R], [sEL, sER, sEL_f0, sER_f0]
    
    def obtain_recursive_list(self, omega, region_type, sE_initial, N_SC, N_junction, 
                            hop_FKs, g0_inv_FKs):
        """Obtain recursive list for given region type"""
        if region_type == 'left_lead':
            sE_initial = [sE * i0 for sE, i0 in zip(sE_initial, [1, 0, 1, 0])]
            range_sets = [0, N_SC, 0]
        elif region_type == 'right_lead':
            sE_initial = [sE * i0 for sE, i0 in zip(sE_initial, [0, 1, 0, 1])]
            range_sets = [N_SC + N_junction, 2 * N_SC + N_junction, N_SC + N_junction]
        elif region_type == 'junction':
            range_sets = [N_SC, N_SC + N_junction, N_SC]
        
        return self.recursive_sweep(omega, sE_initial, range_sets, hop_FKs, g0_inv_FKs)


# ==================== 3. JosephsonJunctionSolver ====================

class JosephsonJunctionSolver:
    """
    约瑟夫森结求解器 - 合并优化版
    支持计算类型: 'CPR', 'DC_IV', 'ABS', 'SPECTRA'
    """
    
    def __init__(self, params, path_manager=None):
        # Ensure sparse config
        params.setdefault('use_sparse', True)
        params.setdefault('sparse_threshold', 0.3)
        params.setdefault('sparse_depth_limit', 5)
        
        self.params = params
        self.path_manager = path_manager
        
        if path_manager:
            self.output_dir = path_manager.get_raw_data_dir()
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = params.get('output_dir', 'results')
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize builders
        self.ham_builder = HamiltonianBuilder(params)
        self.gf_calculator = GreenFunctionCalculator(params)
        
        # Calculation parameters
        self.max_sidebands = params['max_sidebands']
        self.omega_points = params['omega_points']
        self.Vbias_points = params.get('Vbias_points', 300)
        self.B_points = params.get('B_points', 300)
        self.phi_points = params.get('phi_points', 300)
        self.job_parallel = params['job_parallel']
        
        # Lead type configuration (1-C: configurable, default 'infinite')
        self.lead_type = params.get('lead_type', 'infinite')
        
        # Current output configuration (2-B: return_site_currents, default True)
        self.return_site_currents = params.get('return_site_currents', True)
        
        # Adaptive configuration (3-C: multiple methods)
        self.adaptive_iv = params.get('adaptive_iv', False)
        self.adaptive_tol = params.get('adaptive_tol', 0.01)
        self.adaptive_max_N = params.get('adaptive_max_N', 20)
        self.adaptive_method = params.get('adaptive_method', 'advanced')
        self.adaptive_regions = params.get('adaptive_regions', [])
        self.rel_tol = params.get('rel_tol', 0.02)
        self.abs_tol = params.get('abs_tol', 0.01)
        
        # Hamiltonian slice cache
        self.ham_slice_cache = {}
        
        # Common timestamp
        self.common_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Unit conversion
        self.unit_factor_nA = 2 * 1.6e-19 * 1e-3 / (4.13e-15) * 1e9
        
        logging.info(f"JosephsonJunctionSolver initialized")
        logging.info(f"SC: {params['N_SC']} sites, Junction: {params['N_junction']} sites")
        logging.info(f"Lead type: {self.lead_type}, Site currents: {self.return_site_currents}")
        logging.info(f"Adaptive: {self.adaptive_iv}, Method: {self.adaptive_method}")
    
    def _get_base_metadata(self):
        """Return common metadata for all calculations"""
        metadata = {
            "N_SC": self.params['N_SC'],
            "N_junction": self.params['N_junction'],
            "delta": self.params['delta'],
            "alpha": self.params['alpha'],
            "v_tau": self.params['v_tau'],
            "B": self.params['B'],
            "mu": self.params['mu'],
            "mu_lead": self.params['mu_lead'],
            "lead_type": self.lead_type,
            "max_sidebands": self.params['max_sidebands'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "common_timestamp": self.common_timestamp
        }
        
        # Add disorder info
        disorder_type = self.params.get('disorder_type', 'none')
        metadata["disorder_type"] = disorder_type
        metadata["disorder_region"] = self.params.get('disorder_region', 'all')
        metadata["disorder_strength"] = self.params.get('disorder_strength', 1.0)
        
        if disorder_type != 'none':
            if disorder_type == 'gaussian':
                metadata.update({
                    "Vdis": self.params.get('Vdis_gau', 0.0),
                    "decayL": self.params.get('decayL_gau', 50),
                    "Xdis": self.params.get('Xdis_gau', 150)
                })
            elif disorder_type == 'smooth':
                metadata.update({
                    "decayL": self.params.get('decayL_smooth', 50),
                    "Vdis": self.params.get('Vdis_smooth', 0.0),
                    "Vd": self.params.get('Vd_smooth', 0.8)
                })
            elif disorder_type in ['random_typeI', 'random_typeII']:
                metadata.update({
                    "N_imp": self.params.get('N_imp1', 52),
                    "lambda_imp": self.params.get('lambda_imp1', 18.0),
                    "V0": self.params.get('V0_imp1', 0.0),
                    "n_d": self.params.get('Nd_imp2', 10.0),
                    "random_lambda_imp": self.params.get('lambda_imp2', 20.0),
                    "random_V0": self.params.get('V0_imp2', 0.0),
                    "a0": self.params.get('a0', 10.0)
                })
            elif disorder_type == 'from_file':
                metadata.update({
                    "disorder_file": self.params.get('disorder_file', 'disorder_data.txt')
                })
        
        # Add adaptive params
        if self.adaptive_iv:
            metadata.update({
                "adaptive_method": self.adaptive_method,
                "adaptive_tol": self.adaptive_tol,
                "adaptive_max_N": self.adaptive_max_N
            })
        
        return metadata
    
    # ==================== Grid Generation ====================
    
    @staticmethod
    def generate_nonuniform_grid(start, end, dense_start, dense_end,
                                 space_non_dense=0.05, space_ratio=0.4, max_points=300):
        """Generate non-uniform grid with dense region"""
        space_dense = space_non_dense * space_ratio
        
        n_left_segments = max(0, int(np.floor((dense_start - start) / space_non_dense)))
        adjusted_dense_start = start + n_left_segments * space_non_dense
        
        max_dense_end = end - space_non_dense
        
        if max_dense_end > adjusted_dense_start:
            min_segments = max(0, int(np.ceil((dense_end - adjusted_dense_start) / space_dense)))
            max_segments = int(np.floor((max_dense_end - adjusted_dense_start) / space_dense))
            n_dense_segments = min(min_segments, max_segments) if min_segments > 0 else max_segments
            adjusted_dense_end = adjusted_dense_start + n_dense_segments * space_dense
        else:
            n_dense_segments = 0
            adjusted_dense_end = adjusted_dense_start
        
        n_left = n_left_segments
        n_dense = n_dense_segments + 1 if n_dense_segments > 0 else 0
        n_right_segments = max(0, int(np.floor((end - adjusted_dense_end) / space_non_dense)))
        n_right = n_right_segments
        
        points = []
        
        if n_left > 0:
            left_points = np.linspace(start, adjusted_dense_start, n_left + 1, endpoint=False)
            points.extend(left_points)
        
        if n_dense > 0:
            dense_points = np.linspace(adjusted_dense_start, adjusted_dense_end, n_dense)
            points.extend(dense_points)
        
        if n_right > 0:
            right_start = adjusted_dense_end + space_non_dense
            right_points = np.linspace(right_start, 
                                       right_start + n_right * space_non_dense,
                                       n_right + 1, endpoint=True)
            right_points = right_points[right_points <= end]
            points.extend(right_points)
        
        points = np.array(points)
        points.sort()
        
        if abs(points[0] - start) > 1e-10:
            points = np.insert(points, 0, start)
        if abs(points[-1] - end) > 1e-10:
            points = np.append(points, end)
        
        if len(points) > max_points:
            logging.warning(f"Grid points ({len(points)}) exceeds max ({max_points}), truncating")
            points = points[:max_points]
        
        return points
    
    @staticmethod
    def generate_uniform_grid(start, end, spacing=0.05, max_points=300):
        """Generate uniform grid"""
        num_segments = (end - start) / spacing
        n_points = int(np.floor(num_segments)) + 1
        
        if n_points > max_points:
            n_points = max_points
            spacing = (end - start) / (n_points - 1) if n_points > 1 else 0
        
        return np.linspace(start, end, n_points)
    
    # ==================== Core Calculation Methods ====================
    
    def compute_parallel_aux(self, Vari_input, calc_type):
        """
        Parallel computation dispatcher
        Supports: 'CPR', 'DC_IV', 'ABS', 'SPECTRA', 'DC_IV_Bsweep'
        """
        if calc_type == 'CPR':
            phi = Vari_input
            ham = self.ham_builder.construct_hamiltonian(phi)
            
            # CPR: Vbias=0, no Floquet sidebands
            omega_values = self.generate_uniform_grid(
                start=-4.5 * self.params['delta'],
                end=0,
                spacing=0.0025 * self.params['delta'],
                max_points=self.omega_points
            )
            
            # Use finite leads for CPR
            results = Parallel(n_jobs=self.job_parallel[1])(
                delayed(self._compute_current_at_omega_cpr)(
                    omega, ham, phi) for omega in omega_values)
            
            current = np.trapz(np.real(results), omega_values) / (2 * np.pi)
            return current, 0  # 0 sidebands for CPR
        
        elif calc_type == 'DC_IV':
            Vbias = Vari_input
            
            if self.adaptive_iv:
                if self.adaptive_method == 'advanced':
                    return self.adaptive_current_advanced(Vbias)
                elif self.adaptive_method == 'dynamic':
                    return self.adaptive_current_dynamic(Vbias)
                else:
                    current = self.compute_current_at_bias(Vbias)
                    return current, self.max_sidebands
            else:
                current = self.compute_current_at_bias(Vbias)
                return current, self.max_sidebands
        
        elif calc_type == 'ABS':
            phi = Vari_input
            ham = self.ham_builder.construct_hamiltonian(phi)
            
            # ABS: energy grid around gap
            omega_values = self.generate_uniform_grid(
                start=-1.5 * self.params['delta'],
                end=1.5 * self.params['delta'],
                spacing=0.01 * self.params['delta'],
                max_points=self.omega_points
            )
            
            results = Parallel(n_jobs=self.job_parallel[1])(
                delayed(self._compute_dos_at_omega_abs)(
                    omega, ham, phi) for omega in omega_values)
            
            return np.array(results)
        
        elif calc_type == 'SPECTRA':
            B_val = Vari_input
            # Update B field
            self.ham_builder.B = B_val
            ham = self.ham_builder.construct_hamiltonian(0)
            
            omega_values = self.generate_uniform_grid(
                start=-2 * self.params['delta'],
                end=2 * self.params['delta'],
                spacing=0.01 * self.params['delta'],
                max_points=self.omega_points
            )
            
            results = Parallel(n_jobs=self.job_parallel[1])(
                delayed(self._compute_dos_at_omega_spectra)(
                    omega, ham, B_val) for omega in omega_values)
            
            return np.array(results)
        
        elif calc_type == 'DC_IV_Bsweep':
            B, Vbias = Vari_input
            self.ham_builder.B = B
            
            if self.adaptive_iv:
                current, sideband = self.compute_parallel_aux(Vbias, 'DC_IV')
            else:
                current = self.compute_current_at_bias(Vbias)
                sideband = self.max_sidebands
            
            return current, sideband
        
        else:
            raise ValueError(f"Unknown calculation type: {calc_type}")
    
    def _compute_current_at_omega_cpr(self, omega, ham, phi):
        """Compute current at given omega for CPR"""
        # CPR: no Floquet sidebands (max_sidebands=0), finite leads, junction region
        return self._compute_current_at_omega_base(
            omega, 'finite', 'junction', ham, 0, 0.0, 
            self.params['N_SC'], self.params['N_junction'])
    
    def _compute_dos_at_omega_abs(self, omega, ham, phi):
        """Compute DOS at given omega for ABS"""
        max_sidebands = 0
        Vbias = 0
        
        g0_inv_FKs, hop_FKs = self.ham_builder.build_slice_ham(Vbias, ham, max_sidebands)
        
        Gs, sE_initial = self.gf_calculator.compute_self_energies(
            omega, 'finite', ham, max_sidebands, Vbias,
            self.params['N_SC'], self.params['N_junction'], hop_FKs, g0_inv_FKs)
        
        g0_inv, sE4s = self.gf_calculator.obtain_recursive_list(
            omega, 'junction', sE_initial,
            self.params['N_SC'], self.params['N_junction'], hop_FKs, g0_inv_FKs)
        
        recur_depth = self.params['N_junction']
        trace0 = 0
        
        for recur_i in range(recur_depth):
            recur_i_R = recur_depth - 1 - recur_i
            G_nn_ret = la.inv(g0_inv[recur_i] - 
                             sE4s[0][recur_i] - sE4s[1][recur_i_R])
            trace0 += (-1 / np.pi) * np.trace(G_nn_ret).imag
        
        return trace0
    
    def _compute_dos_at_omega_spectra(self, omega, ham, B_val):
        """Compute DOS at given omega for SPECTRA (lead region)"""
        max_sidebands = 0
        Vbias = 0
        
        g0_inv_FKs, hop_FKs = self.ham_builder.build_slice_ham(Vbias, ham, max_sidebands)
        
        Gs, sE_initial = self.gf_calculator.compute_self_energies(
            omega, 'finite', ham, max_sidebands, Vbias,
            self.params['N_SC'], self.params['N_junction'], hop_FKs, g0_inv_FKs)
        
        # For spectra, scan left lead region
        g0_inv, sE4s = self.gf_calculator.obtain_recursive_list(
            omega, 'left_lead', sE_initial,
            self.params['N_SC'], self.params['N_junction'], hop_FKs, g0_inv_FKs)
        
        recur_depth = self.params['N_SC']
        trace0 = 0
        
        for recur_i in range(recur_depth):
            recur_i_R = recur_depth - 1 - recur_i
            G_nn_ret = la.inv(g0_inv[recur_i] - 
                             sE4s[0][recur_i] - sE4s[1][recur_i_R])
            trace0 += (-1 / np.pi) * np.trace(G_nn_ret).imag
        
        return trace0
    
    def _compute_current_at_omega_base(self, omega, lead_type, region_type, 
                                        ham, max_sidebands, Vbias, N_SC, N_junction):
        """Base method for computing current at given omega"""
        g0_inv_FKs, hop_FKs = self.ham_builder.build_slice_ham(Vbias, ham, max_sidebands)
        
        Gs, sE_initial = self.gf_calculator.compute_self_energies(
            omega, lead_type, ham, max_sidebands, Vbias, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        g0_inv, sE4s_leads = self.gf_calculator.obtain_recursive_list(
            omega, region_type, sE_initial, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        recur_depth = N_junction if region_type == 'junction' else N_SC
        
        # Prepare tau_z matrix
        sideN = 2 * max_sidebands + 1
        tau_z_matrix = np.kron(np.eye(sideN), np.kron(self.ham_builder.tau_z, np.eye(2)))
        
        site_currents = []
        
        for recur_i in range(recur_depth):
            recur_i_R = recur_depth - 1 - recur_i
            
            # Convert to dense for final calculation
            gi_inv = g0_inv[recur_i].toarray() if sp.issparse(g0_inv[recur_i]) else g0_inv[recur_i]
            sEL_i = sE4s_leads[0][recur_i].toarray() if sp.issparse(sE4s_leads[0][recur_i]) else sE4s_leads[0][recur_i]
            sER_i = sE4s_leads[1][recur_i_R].toarray() if sp.issparse(sE4s_leads[1][recur_i_R]) else sE4s_leads[1][recur_i_R]
            sEL_less_i = sE4s_leads[2][recur_i].toarray() if sp.issparse(sE4s_leads[2][recur_i]) else sE4s_leads[2][recur_i]
            sER_less_i = sE4s_leads[3][recur_i_R].toarray() if sp.issparse(sE4s_leads[3][recur_i_R]) else sE4s_leads[3][recur_i_R]
            
            G_nn_ret = la.inv(gi_inv - sEL_i - sER_i)
            G_nn_adv = G_nn_ret.conj().T
            
            sE_nn_less = sEL_less_i + sER_less_i
            
            G_nn_less = G_nn_ret @ sE_nn_less @ G_nn_adv
            temp = (G_nn_ret @ sEL_less_i + G_nn_less @ sEL_i.conj().T) @ tau_z_matrix
            
            trace_val = np.trace(temp).real * self.unit_factor_nA
            site_currents.append(trace_val)
        
        return np.array(site_currents) if self.return_site_currents else np.sum(site_currents)
    
    # ==================== DC_IV Specific Methods ====================
    
    def generate_Vbias_vals(self):
        """Generate Vbias values"""
        delta = self.params['delta']
        Vbias_max_ratio = self.params.get('Vbias_max_ratio', 2.0)
        
        if Vbias_max_ratio <= 2.2:
            return self.generate_uniform_grid(0.01, Vbias_max_ratio + 0.01,
                                             0.01, self.Vbias_points)
        else:
            return self.generate_nonuniform_grid(
                start=0.01,
                end=Vbias_max_ratio,
                dense_start=0.12,
                dense_end=1.2,
                space_non_dense=0.04,
                space_ratio=0.25,
                max_points=self.Vbias_points
            )
    
    def compute_current_at_bias(self, Vbias, max_sidebands=None):
        """Compute current at given bias (with caching)"""
        if max_sidebands is None:
            max_sidebands = self.max_sidebands
        
        # Cache check
        cache_key = (Vbias, max_sidebands)
        if cache_key in self.ham_slice_cache:
            g0_inv_FKs, hop_FKs = self.ham_slice_cache[cache_key]
        else:
            ham = self.ham_builder.construct_hamiltonian(0)
            g0_inv_FKs, hop_FKs = self.ham_builder.build_slice_ham(Vbias, ham, max_sidebands)
            self.ham_slice_cache[cache_key] = (g0_inv_FKs, hop_FKs)
        
        N_SC = self.params['N_SC']
        N_junction = self.params['N_junction']
        
        # Omega grid
        if Vbias < 0.12 * self.params['delta']:
            max_points = 21
        else:
            max_points = self.omega_points
        
        omega_values = self.generate_uniform_grid(
            start=0,
            end=Vbias,
            spacing=0.0025 * Vbias,
            max_points=max_points
        )
        
        # Partial function for parallel computation
        compute_func = functools.partial(
            self._compute_current_at_omega_iv,
            g0_inv_FKs=g0_inv_FKs,
            hop_FKs=hop_FKs,
            Vbias=Vbias,
            max_sidebands=max_sidebands
        )
        
        results = Parallel(n_jobs=self.job_parallel[1])(
            delayed(compute_func)(omega) for omega in omega_values)
        
        current_mat = np.real(np.array(results))
        site_currents = np.trapz(current_mat, omega_values, axis=0) / 0.3
        
        return site_currents
    
    def _compute_current_at_omega_iv(self, omega, g0_inv_FKs, hop_FKs, Vbias, max_sidebands):
        """Compute current at omega for IV (uses cached Hamiltonian slices)"""
        N_SC = self.params['N_SC']
        N_junction = self.params['N_junction']
        
        # Use configured lead type
        Gs, sE_initial = self.gf_calculator.compute_self_energies(
            omega, self.lead_type, self.ham_builder.construct_hamiltonian(0),
            max_sidebands, Vbias, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        g0_inv, sE4s_leads = self.gf_calculator.obtain_recursive_list(
            omega, 'junction', sE_initial, N_SC, N_junction, hop_FKs, g0_inv_FKs)
        
        recur_depth = N_junction
        
        sideN = 2 * max_sidebands + 1
        tau_z_matrix = np.kron(np.eye(sideN), np.kron(self.ham_builder.tau_z, np.eye(2)))
        
        site_currents = []
        
        for recur_i in range(recur_depth):
            recur_i_R = recur_depth - 1 - recur_i
            
            gi_inv = g0_inv[recur_i].toarray() if sp.issparse(g0_inv[recur_i]) else g0_inv[recur_i]
            sEL_i = sE4s_leads[0][recur_i].toarray() if sp.issparse(sE4s_leads[0][recur_i]) else sE4s_leads[0][recur_i]
            sER_i = sE4s_leads[1][recur_i_R].toarray() if sp.issparse(sE4s_leads[1][recur_i_R]) else sE4s_leads[1][recur_i_R]
            sEL_less_i = sE4s_leads[2][recur_i].toarray() if sp.issparse(sE4s_leads[2][recur_i]) else sE4s_leads[2][recur_i]
            sER_less_i = sE4s_leads[3][recur_i_R].toarray() if sp.issparse(sE4s_leads[3][recur_i_R]) else sE4s_leads[3][recur_i_R]
            
            G_nn_ret = la.inv(gi_inv - sEL_i - sER_i)
            G_nn_adv = G_nn_ret.conj().T
            
            sE_nn_less = sEL_less_i + sER_less_i
            
            G_nn_less = G_nn_ret @ sE_nn_less @ G_nn_adv
            temp = (G_nn_ret @ sEL_less_i + G_nn_less @ sEL_i.conj().T) @ tau_z_matrix
            
            trace_val = np.trace(temp).real * self.unit_factor_nA
            site_currents.append(trace_val)
        
        return np.array(site_currents)
    
    # ==================== Adaptive Methods (3-C: All three) ====================
    
    def adaptive_current_advanced(self, Vbias):
        """Advanced adaptive strategy (Part 1 style with regions)"""
        delta = self.params['delta']
        ratio = Vbias / delta if delta != 0 else 0
        
        # Find matching region config
        init_N, step_size, max_N = self._get_adaptive_config(ratio)
        
        # Reset Hamiltonian cache for fresh start
        self.ham_builder.prev_sidebands = None
        self.ham_builder.cached_g0_inv_FKs = None
        self.ham_builder.cached_hop_FKs = None
        
        N = init_N
        I_current = self.compute_current_at_bias(Vbias, N)
        history = [(N, I_current)]
        
        for iteration in range(15):
            if N >= max_N:
                logging.warning(f"Reached max sidebands {max_N} @ Vbias={Vbias:.4f}Δ")
                return self._select_best_result(history)
            
            new_N = N + step_size
            logging.debug(f"Iter {iteration}: expanding {N} -> {new_N} @ Vbias={Vbias:.4f}Δ")
            
            I_next = self.compute_current_at_bias(Vbias, new_N)
            history.append((new_N, I_next))
            
            # Calculate change
            if isinstance(I_current, np.ndarray):
                delta_I = np.linalg.norm(I_next - I_current)
                norm_current = np.linalg.norm(I_current) + 1e-6
                rel_delta = delta_I / norm_current
            else:
                delta_I = abs(I_next - I_current)
                rel_delta = delta_I / (abs(I_current) + 1e-6)
            
            logging.debug(f"Sidebands {new_N}: I={I_next}, ΔI={delta_I:.6f}, relΔ={rel_delta:.4f}")
            
            # Convergence check
            if rel_delta < self.rel_tol or delta_I < self.abs_tol:
                logging.info(f"Converged at {new_N} sidebands @ Vbias={Vbias:.4f}Δ")
                return I_next, new_N
            
            I_current = I_next
            N = new_N
        
        logging.warning(f"Max iterations reached @ Vbias={Vbias:.4f}Δ")
        return self._select_best_result(history)
    
    def _get_adaptive_config(self, ratio):
        """Get adaptive configuration for given Vbias/delta ratio"""
        for config in self.adaptive_regions:
            if config[0] <= ratio < config[1]:
                return config[2], config[3], config[4]
        
        # Default config
        if self.adaptive_regions:
            last = self.adaptive_regions[-1]
            if ratio >= last[1]:
                return last[2], last[3], last[4]
        
        return 10, 4, 50
    
    def _select_best_result(self, history):
        """Select best result from non-converged history"""
        if len(history) < 2:
            return history[-1][1], history[-1][0]
        
        changes = []
        for i in range(1, len(history)):
            prev_N, prev_I = history[i-1]
            curr_N, curr_I = history[i]
            
            if isinstance(prev_I, np.ndarray) and isinstance(curr_I, np.ndarray):
                delta = np.linalg.norm(curr_I - prev_I)
                rel_delta = delta / (np.linalg.norm(prev_I) + 1e-6)
            else:
                delta = abs(curr_I - prev_I)
                rel_delta = delta / (abs(prev_I) + 1e-6)
            
            changes.append((rel_delta, delta, i))
        
        best_idx = min(changes, key=lambda x: x[0])[2]
        best_N, best_I = history[best_idx]
        
        logging.info(f"Selected best result: {best_N} sidebands")
        return best_I, best_N
    
    def adaptive_current_dynamic(self, Vbias):
        """Dynamic adaptive strategy (Part 2 style)"""
        delta = self.params['delta']
        
        # Determine parameters based on bias range
        if Vbias < 0.2 * delta:
            base_N = max(20, int(2.0 * delta / (Vbias + 1e-3)))
            large_step = max(5, base_N // 5)
            small_step = max(2, base_N // 10)
            min_step = 1
            max_N = min(150, int(base_N * 30))
            conv_thresh = self.adaptive_tol * 0.5
        elif Vbias < 0.5 * delta:
            base_N = max(10, int(1.5 * delta / (Vbias + 1e-3)))
            large_step = max(3, base_N // 5)
            small_step = 1
            min_step = 1
            max_N = min(100, int(base_N * 30))
            conv_thresh = self.adaptive_tol
        else:
            base_N = max(5, int(0.8 * delta / (Vbias + 1e-3)))
            large_step = 2
            small_step = 1
            min_step = 1
            max_N = min(50, int(base_N * 30))
            conv_thresh = self.adaptive_tol * 1.5
        
        large_step_thresh = 0.1
        
        N = base_N
        prev_I = None
        prev_prev_I = None
        
        for attempt in range(20):
            try:
                I = self.compute_current_at_bias(Vbias, N)
            except Exception as e:
                logging.error(f"Calculation failed @ Vbias={Vbias:.4f}, N={N}: {e}")
                N = max(base_N, N - large_step)
                continue
            
            if prev_I is None:
                prev_I = I
                N += large_step
                continue
            
            delta_I = abs(I - prev_I)
            rel_delta = delta_I / (abs(prev_I) + 1e-6)
            
            # Convergence check
            if rel_delta < conv_thresh:
                if prev_prev_I is not None:
                    prev_delta = abs(prev_I - prev_prev_I)
                    if prev_delta < rel_delta * 2:
                        return I, N
                else:
                    return I, N
            
            # Dynamic step adjustment
            if rel_delta > large_step_thresh:
                step = large_step
            elif rel_delta > conv_thresh * 5:
                step = small_step
            else:
                step = min_step
            
            prev_prev_I = prev_I
            prev_I = I
            N += step
            
            if N > max_N:
                logging.warning(f"Reached max sidebands {max_N} @ Vbias={Vbias:.4f}")
                return I, min(N, max_N)
        
        logging.warning(f"Max attempts reached @ Vbias={Vbias:.4f}")
        return I, N
    
    # ==================== High-Level Calculation Wrappers ====================
    
    def compute_cpr(self):
        """Compute Current-Phase Relation"""
        start_time = time.time()
        
        phi_vals = self.generate_nonuniform_grid(
            start=0, end=2 * np.pi,
            dense_start=0.4 * np.pi, dense_end=1.6 * np.pi,
            space_non_dense=0.05, space_ratio=0.4,
            max_points=self.phi_points
        )
        
        logging.info(f"Starting CPR calculation, {len(phi_vals)} phases")
        results = Parallel(n_jobs=self.job_parallel[0])(
            delayed(self.compute_parallel_aux)(phi, 'CPR') for phi in phi_vals
        )
        
        currents = [res[0] for res in results]
        
        metadata = self._get_base_metadata()
        metadata.update({
            "calculation": "CPR",
            "duration": time.time() - start_time,
            "phi_points": len(phi_vals)
        })
        
        logging.info(f"CPR completed in {metadata['duration']:.2f}s")
        return metadata, phi_vals, currents
    
    def compute_dc_iv(self):
        """Compute DC I-V characteristic"""
        start_time = time.time()
        delta = self.params['delta']
        
        Vbias_vals = delta * self.generate_Vbias_vals()
        
        logging.info(f"Starting DC IV calculation, {len(Vbias_vals)} points")
        results = Parallel(n_jobs=self.job_parallel[0])(
            delayed(self.compute_parallel_aux)(Vbias, 'DC_IV') for Vbias in Vbias_vals
        )
        
        currents_list = [res[0] for res in results]
        sideband_Ns = [res[1] for res in results]
        
        # Handle both site-currents and scalar currents
        first_current = currents_list[0]
        if isinstance(first_current, np.ndarray):
            currents = np.array(currents_list)
            logging.info(f"Current data shape: {currents.shape} (site-resolved)")
        else:
            currents = np.array(currents_list)
        
        metadata = self._get_base_metadata()
        metadata.update({
            "calculation": "DC_IV",
            "duration": time.time() - start_time,
            "Vbias_points": len(Vbias_vals),
            "current_data_type": "array_sites" if isinstance(first_current, np.ndarray) else "scalar"
        })
        
        logging.info(f"DC IV completed in {metadata['duration']:.2f}s")
        return metadata, Vbias_vals, currents, sideband_Ns
    
    def compute_dc_iv_Bsweep(self):
        """Compute DC IV with B-field sweep"""
        start_time = time.time()
        Vbias = self.params.get('fixed_Vbias', 0.1)
        B_max = self.params.get('B_max', 2.0)
        
        B_vals = self.generate_uniform_grid(0, B_max, 0.02, self.B_points)
        
        logging.info(f"Starting DC IV B-sweep @ Vbias={Vbias}meV, {len(B_vals)} points")
        
        # Prepare inputs
        inputs = [(B, Vbias) for B in B_vals]
        
        results = Parallel(n_jobs=self.job_parallel[0])(
            delayed(self.compute_parallel_aux)(inp, 'DC_IV_Bsweep') for inp in inputs
        )
        
        currents_list = [res[0] for res in results]
        sideband_Ns = [res[1] for res in results]
        
        first_current = currents_list[0]
        if isinstance(first_current, np.ndarray):
            currents = np.array(currents_list)
        else:
            currents = np.array(currents_list)
        
        metadata = self._get_base_metadata()
        metadata.update({
            "calculation": "DC_IV_Bsweep",
            "duration": time.time() - start_time,
            "fixed_Vbias": Vbias,
            "B_points": len(B_vals),
            "B_max": B_max
        })
        
        logging.info(f"DC IV B-sweep completed in {metadata['duration']:.2f}s")
        return metadata, B_vals, currents, sideband_Ns
    
    def compute_abs(self):
        """Compute Andreev Bound States spectrum"""
        start_time = time.time()
        
        phi_vals = self.generate_nonuniform_grid(
            start=0, end=2 * np.pi,
            dense_start=0.4 * np.pi, dense_end=1.6 * np.pi,
            space_non_dense=0.05, space_ratio=0.4,
            max_points=self.phi_points
        )
        
        logging.info(f"Starting ABS calculation, {len(phi_vals)} phases")
        abs_dos = Parallel(n_jobs=self.job_parallel[0])(
            delayed(self.compute_parallel_aux)(phi, 'ABS') for phi in phi_vals
        )
        
        # Energy values used in calculation
        E_vals = self.generate_uniform_grid(
            start=-1.5 * self.params['delta'],
            end=1.5 * self.params['delta'],
            spacing=0.01 * self.params['delta'],
            max_points=self.omega_points
        )
        
        metadata = self._get_base_metadata()
        metadata.update({
            "calculation": "ABS",
            "duration": time.time() - start_time,
            "phi_points": len(phi_vals),
            "omega_points": len(E_vals)
        })
        
        logging.info(f"ABS completed in {metadata['duration']:.2f}s")
        return metadata, phi_vals, E_vals, np.array(abs_dos)
    
    def compute_spectra(self):
        """Compute lead energy spectra"""
        start_time = time.time()
        
        B_vals = self.generate_uniform_grid(
            start=0, end=2, spacing=0.02, max_points=self.B_points)
        
        E_vals = self.generate_uniform_grid(
            start=-2 * self.params['delta'],
            end=2 * self.params['delta'],
            spacing=0.01 * self.params['delta'],
            max_points=self.omega_points
        )
        
        logging.info(f"Starting spectra calculation, {len(B_vals)} B-points")
        spectra_dos = Parallel(n_jobs=self.job_parallel[0])(
            delayed(self.compute_parallel_aux)(B, 'SPECTRA') for B in B_vals
        )
        
        metadata = self._get_base_metadata()
        metadata.update({
            "calculation": "SPECTRA",
            "duration": time.time() - start_time,
            "B_points": len(B_vals),
            "omega_points": len(E_vals)
        })
        
        logging.info(f"Spectra completed in {metadata['duration']:.2f}s")
        return metadata, B_vals, E_vals, np.array(spectra_dos)
    
    # ==================== I/O and Plotting ====================
    
    def run_calculation(self, calc_type):
        """
        Run specified calculation
        calc_type: 'CPR', 'DC_IV', 'ABS', 'SPECTRA', 'DC_IV_Bsweep'
        """
        # Save disorder data if present
        if self.params.get('disorder_type', 'none') not in ['none', 'from_file']:
            self.save_disorder_data()
        
        if self.path_manager:
            self.path_manager.save_task_metadata(self.params)
        
        # Dispatch to appropriate method
        if calc_type == 'CPR':
            metadata, x_vals, y_vals = self.compute_cpr()
            data = (x_vals, y_vals)
        elif calc_type == 'DC_IV':
            metadata, x_vals, y_vals, sidebands = self.compute_dc_iv()
            data = (x_vals, y_vals, sidebands)
        elif calc_type == 'DC_IV_Bsweep':
            metadata, x_vals, y_vals, sidebands = self.compute_dc_iv_Bsweep()
            data = (x_vals, y_vals, sidebands)
        elif calc_type == 'ABS':
            metadata, x_vals, E_vals, dos = self.compute_abs()
            data = (x_vals, E_vals, dos)
        elif calc_type == 'SPECTRA':
            metadata, x_vals, E_vals, dos = self.compute_spectra()
            data = (x_vals, E_vals, dos)
        else:
            raise ValueError(f"Invalid calculation type: {calc_type}")
        
        # Save and plot
        filepath = self.save_results(metadata, data)
        
        if self.path_manager:
            plot_path = JJPlotter.plot_result(metadata, data, self.path_manager.get_plots_dir())
            return filepath, plot_path
        
        return filepath, None
    
    def save_results(self, metadata, data):
        """Save calculation results to CSV"""
        calc_type = metadata["calculation"]
        filename = f"{calc_type}_{self.common_timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Build metadata lines
        metadata_lines = ["# Calculation Metadata"]
        for key, value in metadata.items():
            metadata_lines.append(f"# {key}: {value}")
        
        # Create DataFrame based on calculation type
        if calc_type == 'CPR':
            phi, current = data
            df = pd.DataFrame({
                'Phase (rad)': phi,
                'Current (nA)': current
            })
        
        elif calc_type in ['DC_IV', 'DC_IV_Bsweep']:
            x_vals, currents, sidebands = data
            
            if isinstance(currents, np.ndarray) and currents.ndim == 2:
                # Site-resolved currents
                n_sites = currents.shape[1]
                df_data = {'Bias Voltage (meV)' if calc_type == 'DC_IV' else 'Magnetic Field': x_vals}
                for i in range(n_sites):
                    df_data[f'Current_site_{i}'] = currents[:, i]
                df_data['Current_total'] = currents.sum(axis=1)
                df_data['SidebandN'] = sidebands
                df = pd.DataFrame(df_data)
            else:
                df = pd.DataFrame({
                    'Bias Voltage (meV)' if calc_type == 'DC_IV' else 'Magnetic Field': x_vals,
                    'Current (nA)': currents,
                    'SidebandN': sidebands
                })
        
        elif calc_type in ['ABS', 'SPECTRA']:
            x_vals, E_vals, dos = data
            x_label = 'Phase (rad)' if calc_type == 'ABS' else 'Magnetic Field'
            X, Y = np.meshgrid(x_vals, E_vals, indexing='ij')
            df = pd.DataFrame({
                x_label: X.flatten(),
                'Energy (meV)': Y.flatten(),
                'DOS': dos.flatten()
            })
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(metadata_lines))
            f.write("\n")
            df.to_csv(f, index=False)
        
        logging.info(f"Results saved: {filepath}")
        return filepath
    
    def save_disorder_data(self):
        """Save disorder distribution data"""
        if self.params.get('disorder_type', 'none') in ['none', 'from_file']:
            return {"csv_file": None, "plot_file": None}
        
        disorder_dir = os.path.join(self.output_dir, "disorder_data")
        os.makedirs(disorder_dir, exist_ok=True)
        
        # Save plot
        plot_file = os.path.join(disorder_dir, f"disorder_plot_{self.common_timestamp}.png")
        plt.figure(figsize=(12, 6))
        plt.plot(self.ham_builder.disorder_distribution, 'b-', linewidth=1.5)
        plt.axvline(x=self.params['N_SC'], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=self.params['N_SC'] + self.params['N_junction'], color='r', linestyle='--', alpha=0.5)
        plt.title(f"Disorder ({self.params.get('disorder_type')}, {self.params.get('disorder_region', 'all')})")
        plt.xlabel('Site Index')
        plt.ylabel('Potential (meV)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        # Save CSV
        csv_file = os.path.join(disorder_dir, f"disorder_data_{self.common_timestamp}.csv")
        df = pd.DataFrame({
            'site_index': np.arange(len(self.ham_builder.disorder_distribution)),
            'disorder_value': self.ham_builder.disorder_distribution
        })
        df.to_csv(csv_file, index=False)
        
        logging.info(f"Disorder data saved: {csv_file}, {plot_file}")
        return {"csv_file": csv_file, "plot_file": plot_file}


# ==================== 4. JJPlotter ====================

class JJPlotter:
    """Plotting utilities for Josephson junction calculations"""
    
    @staticmethod
    def plot_result(metadata, data, output_dir):
        """Plot calculation results"""
        calc_type = metadata["calculation"]
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{calc_type}_plot_{metadata.get('common_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}.png"
        filepath = os.path.join(output_dir, filename)
        
        if calc_type == 'CPR':
            phi, current = data
            plt.figure(figsize=(10, 6))
            plt.plot(phi / np.pi, current, 'o-', lw=2, color='b')
            plt.xlabel('Phase Difference ($\\phi/\\pi$)', fontsize=14)
            plt.ylabel('Current (nA)', fontsize=14)
            plt.title('Current-Phase Relation', fontsize=16)
            plt.grid(True)
        
        elif calc_type == 'DC_IV':
            Vbias, currents, sidebands = data
            
            if isinstance(currents, np.ndarray) and currents.ndim == 2:
                # Multi-panel for site-resolved
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                
                # Total current
                total_current = currents.sum(axis=1)
                axes[0].plot(Vbias, total_current, 'r-', lw=2)
                axes[0].set_ylabel('Total Current (nA)', fontsize=12)
                axes[0].set_title('DC I-V Characteristic', fontsize=14)
                axes[0].grid(True)
                
                # Site current heatmap
                X, Y = np.meshgrid(Vbias, np.arange(currents.shape[1]))
                pc = axes[1].pcolormesh(X, Y, currents.T, shading='auto', cmap='viridis')
                axes[1].set_ylabel('Site Index', fontsize=12)
                plt.colorbar(pc, ax=axes[1], label='Current (nA)')
                
                # Sample sites
                n_sites = currents.shape[1]
                selected = np.linspace(0, n_sites-1, min(5, n_sites), dtype=int)
                for idx in selected:
                    axes[2].plot(Vbias, currents[:, idx], '-', label=f'Site {idx}', lw=1.5)
                axes[2].set_xlabel('Bias Voltage (meV)', fontsize=12)
                axes[2].set_ylabel('Current (nA)', fontsize=12)
                axes[2].legend()
                axes[2].grid(True)
                
                # Sideband info on top plot
                ax_twin = axes[0].twinx()
                ax_twin.plot(Vbias, sidebands, 'b--', alpha=0.7, lw=1.5)
                ax_twin.set_ylabel('Sideband Number', fontsize=10, color='b')
            else:
                # Simple IV curve with sidebands
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
                ax1.plot(Vbias, currents, 'r-', lw=2)
                ax1.set_ylabel('Current (nA)', fontsize=14)
                ax1.set_title('DC I-V Characteristic', fontsize=16)
                ax1.grid(True)
                
                ax2.plot(Vbias, sidebands, 'b-', lw=2)
                ax2.set_xlabel('Bias Voltage (meV)', fontsize=14)
                ax2.set_ylabel('Sideband Number', fontsize=14)
                ax2.grid(True)
        
        elif calc_type == 'DC_IV_Bsweep':
            B_vals, currents, sidebands = data
            
            if isinstance(currents, np.ndarray) and currents.ndim == 2:
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                
                total_current = currents.sum(axis=1)
                axes[0].plot(B_vals, total_current, 'r-', lw=2, marker='o')
                axes[0].set_ylabel('Total Current (nA)', fontsize=12)
                axes[0].set_title(f'DC I-V B-sweep @ Vbias={metadata.get("fixed_Vbias")}meV', fontsize=14)
                axes[0].grid(True)
                
                X, Y = np.meshgrid(B_vals, np.arange(currents.shape[1]))
                pc = axes[1].pcolormesh(X, Y, currents.T, shading='auto', cmap='viridis')
                axes[1].set_ylabel('Site Index', fontsize=12)
                plt.colorbar(pc, ax=axes[1], label='Current (nA)')
                
                n_sites = currents.shape[1]
                selected = np.linspace(0, n_sites-1, min(5, n_sites), dtype=int)
                for idx in selected:
                    axes[2].plot(B_vals, currents[:, idx], '-o', label=f'Site {idx}', lw=1.5, markersize=4)
                axes[2].set_xlabel('Magnetic Field (Zeeman energy)', fontsize=12)
                axes[2].set_ylabel('Current (nA)', fontsize=12)
                axes[2].legend()
                axes[2].grid(True)
                
                ax_twin = axes[0].twinx()
                ax_twin.plot(B_vals, sidebands, 'b--', alpha=0.7, lw=1.5)
                ax_twin.set_ylabel('Sideband Number', fontsize=10, color='b')
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
                ax1.plot(B_vals, currents, 'r-', lw=2, marker='o')
                ax1.set_ylabel('Current (nA)', fontsize=14)
                ax1.set_title(f'DC I-V B-sweep @ Vbias={metadata.get("fixed_Vbias")}meV', fontsize=16)
                ax1.grid(True)
                
                ax2.plot(B_vals, sidebands, 'b-', lw=2)
                ax2.set_xlabel('Magnetic Field (Zeeman energy)', fontsize=14)
                ax2.set_ylabel('Sideband Number', fontsize=14)
                ax2.grid(True)
        
        elif calc_type == 'ABS':
            phi, E_vals, dos = data
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(phi / np.pi, E_vals, dos.T, shading='auto', 
                          vmin=-0.5, vmax=0.5, cmap='seismic')
            plt.xlabel('Phase Difference ($\\phi/\\pi$)', fontsize=14)
            plt.ylabel('Energy (meV)', fontsize=14)
            plt.title('Andreev Bound States Spectrum', fontsize=16)
            plt.colorbar(label='DOS')
        
        elif calc_type == 'SPECTRA':
            B_vals, E_vals, dos = data
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(B_vals, E_vals, dos.T, shading='auto',
                          vmin=-20, vmax=20, cmap='seismic')
            plt.xlabel('Magnetic Field (Zeeman energy)', fontsize=14)
            plt.ylabel('Energy (meV)', fontsize=14)
            plt.title('Lead Energy Spectra', fontsize=16)
            plt.colorbar(label='DOS')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Plot saved: {filepath}")
        return filepath


# ==================== 5. PathManager ====================

class PathManager:
    """File path management for simulation tasks"""
    
    def __init__(self, base_dir="results", task_name=None):
        self.base_dir = base_dir
        self.task_name = task_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task_dir = os.path.join(self.base_dir, self.task_name)
        self._create_directories()
    
    def _create_directories(self):
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.get_raw_data_dir(), exist_ok=True)
        os.makedirs(self.get_plots_dir(), exist_ok=True)
    
    def get_raw_data_dir(self):
        return os.path.join(self.task_dir, "raw_data")
    
    def get_plots_dir(self):
        return os.path.join(self.task_dir, "plots")
    
    def get_task_metadata_path(self):
        return os.path.join(self.task_dir, "task_metadata.json")
    
    def save_task_metadata(self, params):
        metadata = {
            "task_name": self.task_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": self._filter_params(params)
        }
        
        with open(self.get_task_metadata_path(), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return self.get_task_metadata_path()
    
    def _filter_params(self, params):
        filtered = deepcopy(params)
        if 'disorder_distribution' in filtered:
            del filtered['disorder_distribution']
        return filtered


# ==================== 6. JJResultsProcessor (from Part 2) ====================

class JJResultsProcessor:
    """Batch parameter sweep processor (from Part 2)"""
    
    def __init__(self, base_dir="results", outer_parallel=4):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.outer_parallel = outer_parallel
    
    def _run_single_parameter_value(self, params, calc_type, param_name, param_value):
        """Execute single parameter value calculation"""
        local_params = deepcopy(params)
        local_params[param_name] = param_value
        
        # Create temporary path manager
        path_mgr = PathManager(base_dir=self.base_dir, 
                              task_name=f"temp_{param_name}_{param_value}_{datetime.now().strftime('%H%M%S')}")
        
        solver = JosephsonJunctionSolver(local_params, path_mgr)
        results = solver.run_calculation(calc_type)
        metadata = solver._get_base_metadata()
        metadata[param_name] = param_value
        
        return results, metadata
    
    def run_single_param_sweep(self, params, param_name, param_values, calc_type, **kwargs):
        """
        Parallel single parameter sweep
        
        Parameters:
        -----------
        params : dict
            Base parameters
        param_name : str
            Parameter to sweep
        param_values : list
            Values to scan
        calc_type : str
            'CPR', 'DC_IV', 'ABS', 'SPECTRA'
        """
        params = deepcopy(params)
        params.update(kwargs)
        
        all_results = []
        total_values = len(param_values)
        base_metadata = None
        
        # Determine result headers
        headers = {
            'CPR': ('Phase (rad)', 'Current (nA)'),
            'DC_IV': ('Bias Voltage (meV)', 'Current (nA)'),
            'ABS': ('Phase (rad)', 'DOS'),
            'SPECTRA': ('Magnetic Field', 'DOS')
        }
        x_header, result_header = headers.get(calc_type, ('X', 'Y'))
        
        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.outer_parallel) as executor:
            futures = {
                executor.submit(self._run_single_parameter_value, 
                              params, calc_type, param_name, value): value 
                for value in param_values
            }
            
            results_dict = {}
            for i, future in enumerate(as_completed(futures)):
                param_value = futures[future]
                try:
                    results, metadata = future.result()
                    results_dict[param_value] = (results, metadata)
                    
                    if base_metadata is None:
                        base_metadata = metadata
                    
                    print(f"Completed ({i+1}/{total_values}): {param_name}={param_value}")
                except Exception as e:
                    print(f"Error for {param_name}={param_value}: {e}")
        
        # Process results in order
        if base_metadata:
            base_metadata.update({
                "calculation": calc_type,
                "sweep_parameter": param_name,
                "sweep_values": list(param_values)
            })
        
        for value in param_values:
            if value in results_dict:
                (filepath, plotpath), meta = results_dict[value]
                # Load data from saved file
                df = pd.read_csv(filepath, comment='#')
                
                # Append parameter column
                df.insert(0, param_name, value)
                all_results.append(df)
        
        # Combine and save
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{calc_type}_{param_name}_sweep_{timestamp}.csv"
            filepath = os.path.join(self.base_dir, filename)
            
            # Add metadata header
            metadata_str = "# Single Parameter Sweep Metadata\n"
            if base_metadata:
                for key, value in base_metadata.items():
                    metadata_str += f"# {key}: {value}\n"
            
            with open(filepath, 'w') as f:
                f.write(metadata_str)
                combined_df.to_csv(f, index=False)
            
            # Plot
            plot_file = self._plot_sweep(combined_df, param_name, x_header, result_header, calc_type)
            
            return filepath, plot_file
        
        return None, None
    
    def _plot_sweep(self, df, param_name, x_header, result_header, calc_type):
        """Plot parameter sweep results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.base_dir, f"{calc_type}_sweep_plot_{timestamp}.png")
        
        unique_params = df[param_name].unique()
        
        if calc_type in ['CPR', 'DC_IV']:
            plt.figure(figsize=(10, 6))
            for val in unique_params:
                subset = df[df[param_name] == val]
                plt.plot(subset[x_header].values, subset[result_header].values,
                        'o-', label=f"{param_name}={val:.4f}", lw=1.5)
            plt.xlabel(x_header)
            plt.ylabel(result_header)
            plt.legend()
            plt.title(f"{calc_type} - {param_name} Sweep")
            plt.grid(True)
        
        elif calc_type in ['ABS', 'SPECTRA']:
            # Multi-panel for spectra
            n_cols = min(3, len(unique_params))
            n_rows = (len(unique_params) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), squeeze=False)
            fig.suptitle(f"{calc_type} Spectrum - {param_name} Sweep", fontsize=16)
            
            for i, val in enumerate(unique_params):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                subset = df[df[param_name] == val]
                x_vals = subset[x_header].unique()
                E_vals = subset['Energy (meV)'].unique()
                
                Z = np.zeros((len(x_vals), len(E_vals)))
                for xi, x in enumerate(x_vals):
                    for ei, e in enumerate(E_vals):
                        match = subset[(subset[x_header] == x) & (subset['Energy (meV)'] == e)]
                        if not match.empty:
                            Z[xi, ei] = match[result_header].values[0]
                
                im = ax.pcolormesh(x_vals, E_vals, Z.T, shading='auto')
                ax.set_xlabel(x_header)
                ax.set_ylabel('Energy (meV)')
                ax.set_title(f"{param_name} = {val:.4f}")
                plt.colorbar(im, ax=ax, label=result_header)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        return plot_file


# ==================== 7. Parameter Utilities ====================

def get_default_parameters():
    """Get default simulation parameters"""
    return {
        # System parameters
        'N_SC': 150,
        'N_junction': 2,
        't': 12.7,
        'delta': 0.3,
        'mu': 0.0,
        'mu_lead': 0.0,
        'B': 0.0,
        'alpha': 4.0,
        'v_tau': 0.85,
        
        # Floquet parameters
        'max_sidebands': 10,
        'Vbias_max_ratio': 3.0,
        
        # Grid parameters
        'omega_points': 2000,
        'phi_points': 300,
        'B_points': 300,
        'Vbias_points': 300,
        
        # Calculation parameters
        'recursion_depth': 100,
        'eta': 1e-3,
        'job_parallel': [4, 1],  # [outer, inner]
        
        # Lead configuration (1-C)
        'lead_type': 'infinite',  # 'infinite' or 'finite'
        
        # Current output (2-B)
        'return_site_currents': True,
        
        # Sparse matrix (Part 1 optimized)
        'use_sparse': True,
        'sparse_threshold': 0.3,
        'sparse_depth_limit': 5,
        
        # Disorder parameters (Part 2 features)
        'disorder_type': 'none',
        'disorder_region': 'all',  # 'all', 'left_lead', 'junction', 'right_lead', 'all_leads'
        'disorder_strength': 1.0,
        'disorder_file': 'disorder_data.txt',
        
        # Gaussian disorder
        'Vdis_gau': 0.0,
        'decayL_gau': 50,
        'Xdis_gau': 150,
        
        # Smooth disorder
        'decayL_smooth': 50,
        'Vdis_smooth': 0.0,
        'Vd_smooth': 0.8,
        
        # Random type I
        'N_imp1': 52,
        'lambda_imp1': 18.0,
        'V0_imp1': 0.0,
        
        # Random type II
        'Nd_imp2': 10.0,
        'lambda_imp2': 20.0,
        'V0_imp2': 0.0,
        'a0': 10.0,
        
        # Adaptive parameters (3-C: all methods)
        'adaptive_iv': False,
        'adaptive_method': 'advanced',  # 'advanced', 'dynamic', 'basic'
        'adaptive_tol': 0.01,
        'adaptive_max_N': 20,
        'rel_tol': 0.02,
        'abs_tol': 0.01,
        'adaptive_regions': [
            (0.0, 0.025, 50, 10, 220),
            (0.025, 0.05, 40, 2, 80),
            (0.05, 0.1, 38, 2, 70),
            (0.1, 0.2, 20, 2, 50),
            (0.2, 0.4, 10, 2, 30),
            (0.4, 0.7, 8, 1, 20),
            (0.7, 1.0, 6, 1, 12),
            (1.0, 6.0, 4, 1, 10)
        ],
        
        # B-sweep parameters
        'fixed_Vbias': 0.1,
        'B_max': 2.0,
        
        # Output
        'output_dir': 'results',
        'mid_site_i': None  # Auto-set to N_SC + N_junction//2 if None
    }


def update_parameters(user_params=None):
    """Update default parameters with user values"""
    params = get_default_parameters()
    updated = deepcopy(params)
    
    if user_params:
        for key, value in user_params.items():
            # Special handling for nested structures
            if key == 'adaptive_regions' and isinstance(value, list):
                updated[key] = deepcopy(value)
            elif key in ['disorder_type', 'disorder_region', 'lead_type', 'adaptive_method']:
                # Validate enum-like parameters
                updated[key] = value
            elif key in updated:
                updated[key] = value
            else:
                logging.warning(f"Ignoring unknown parameter: {key}")
    
    # Auto-set mid_site_i if not provided
    if updated['mid_site_i'] is None:
        updated['mid_site_i'] = updated['N_SC'] + updated['N_junction'] // 2
    
    # Log final parameters
    logging.info("Final simulation parameters:")
    for key, value in updated.items():
        if key != 'adaptive_regions':  # Skip long list
            logging.info(f"  {key}: {value}")
    
    return updated


# ==================== 8. Example Usage ====================

def example_cpr_calculation():
    """Example: CPR calculation"""
    params = update_parameters({
        'N_SC': 150,
        'N_junction': 2,
        'v_tau': 0.85,
        'B': 0.0,
        'max_sidebands': 0,  # No Floquet for CPR
        'omega_points': 1801,
        'phi_points': 242,
        'job_parallel': [4, 1],
        'lead_type': 'finite',  # Finite leads for CPR
        'disorder_type': 'none'
    })
    
    path_mgr = PathManager(base_dir="results", task_name="cpr_example")
    solver = JosephsonJunctionSolver(params, path_mgr)
    
    filepath, plotpath = solver.run_calculation('CPR')
    print(f"Results: {filepath}, Plot: {plotpath}")
    
    return solver


def example_dc_iv_calculation():
    """Example: DC IV calculation with adaptive sidebands"""
    params = update_parameters({
        'N_SC': 150,
        'N_junction': 5,
        'v_tau': 0.882,
        'B': 0.0,
        'max_sidebands': 5,
        'Vbias_max_ratio': 5.0,
        'Vbias_points': 100,
        'omega_points': 501,
        'job_parallel': [4, 1],
        'lead_type': 'infinite',
        'adaptive_iv': True,
        'adaptive_method': 'advanced',
        'return_site_currents': True,
        'disorder_type': 'none'
    })
    
    path_mgr = PathManager(base_dir="results", task_name="dc_iv_example")
    solver = JosephsonJunctionSolver(params, path_mgr)
    
    filepath, plotpath = solver.run_calculation('DC_IV')
    print(f"Results: {filepath}, Plot: {plotpath}")
    
    return solver


def example_parameter_sweep():
    """Example: Parameter sweep using JJResultsProcessor"""
    params = update_parameters({
        'N_SC': 150,
        'N_junction': 2,
        'v_tau': 0.338,
        'max_sidebands': 5,
        'omega_points': 5000,
        'phi_points': 300,
        'job_parallel': [1, 1]  # Inner parallel disabled for sweep
    })
    
    processor = JJResultsProcessor(base_dir="sweep_results", outer_parallel=4)
    
    # Sweep B field for CPR
    b_values = np.linspace(0, 2, 21)
    data_file, plot_file = processor.run_single_param_sweep(
        params,
        param_name="B",
        param_values=b_values,
        calc_type='CPR'
    )
    
    print(f"Sweep data: {data_file}, Plot: {plot_file}")
    return processor


if __name__ == "__main__":
    # Run example
    print("Josephson Junction Solver - Merged Optimized Version")
    print("=" * 60)
    
    # Uncomment to run examples:
    # example_cpr_calculation()
    # example_dc_iv_calculation()
    # example_parameter_sweep()
    
    print("\nAvailable calculation types:")
    print("  - 'CPR': Current-Phase Relation")
    print("  - 'DC_IV': DC I-V characteristic")
    print("  - 'DC_IV_Bsweep': I-V with B-field sweep")
    print("  - 'ABS': Andreev Bound States")
    print("  - 'SPECTRA': Lead energy spectra")
    print("\nKey features:")
    print("  - Sparse/dense matrix auto-selection")
    print("  - Incremental Floquet Hamiltonian expansion")
    print("  - Multi-level caching")
    print("  - Three adaptive strategies: 'advanced', 'dynamic', 'basic'")
    print("  - Site-resolved current distribution")
    print("  - Disorder region selection")
