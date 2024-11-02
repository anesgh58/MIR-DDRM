import torch
import numpy as np
import cupy as cp


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        output = self.U(singulars * temp[:, :singulars.shape[0]])
        print('singulars',singulars)
        print('temp',temp)
        import scipy.io
        scipy.io.savemat('H_values.mat',
                         {'Vt': temp.detach().cpu().numpy(), 'singulars': singulars.detach().cpu().numpy(),
                          'output': output.detach().cpu().numpy()})
        return output

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        inv_singulars = torch.where(singulars != 0, 1.0 / singulars, torch.tensor(float('inf')))

        # Replace any infinities with a large value (e.g., 1000)
        inv_singulars = torch.where(torch.isinf(inv_singulars), torch.tensor(1000.0), inv_singulars)

        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * inv_singulars
        output = self.V(self.add_zeros(temp))
        import scipy.io
        scipy.io.savemat('H_values.mat',
                         {'Vt': temp.detach().cpu().numpy(), 'singulars': inv_singulars.detach().cpu().numpy(),
                          'output': output.detach().cpu().numpy()})
        
        return self.V(self.add_zeros(temp))


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                                      1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(channels * img_dim ** 2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim ** 2) if i not in missing_indices]).to(
            device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim ** 2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
        
class deconvolution_BCCB(H_functions):
    def __init__(self, kernel, device):
        self.kernel = kernel
        self.device = device
        self.channels = 3
        self.dim = 512

    def to_tensor(self, vec):
        # Convert to tensor if it's a numpy array
        if isinstance(vec, np.ndarray):
            if np.iscomplexobj(vec):
                return torch.tensor(vec, dtype=torch.complex128, device=self.device)
            else:
                return torch.tensor(vec, dtype=torch.float32, device=self.device)  # Move to specified device
        print('isinstance:', isinstance(vec, np.ndarray))
        print(vec.dtype)
        return vec.to(self.device)  # Already a tensor, move to specified device

    def to_numpy(self, vec):
        # Convert to numpy array if it's a tensor
        if isinstance(vec, torch.Tensor):
            return vec.cpu().numpy()  # Move to CPU and convert to numpy
        return vec  # Already a numpy array

    def V(self, vec):
        # Ensure vec is a tensor
        # vec = self.to_tensor(vec)

        # Apply IFFT2 on each channel separately for each batch
        ifft_result = np.zeros_like(self.to_numpy(vec), dtype=np.complex128)

        for b in range(vec.shape[0]):  # Loop over each batch
            for c in range(self.channels):  # Loop over each channel
                # Apply IFFT2 on each 2D channel
                ifft_result[b, c, :, :] = np.fft.ifft2(self.to_numpy(vec)[b, c, :, :])
        
        ifft_result = self.to_tensor(ifft_result)
        # Colomn wise vectorization
        return ifft_result.reshape(vec.shape[0],-1)

    def Vt(self, vec):
        # Ensure vec is a tensor
        # vec = self.to_tensor(vec)

        # Apply IFFT2 on each channel separately for each batch
        fft_result = np.zeros_like(self.to_numpy(vec), dtype=np.complex128)

        for b in range(vec.shape[0]):  # Loop over each batch
            for c in range(self.channels):  # Loop over each channel
                # Apply IFFT2 on each 2D channel
                fft_result[b, c, :, :] = np.fft.fft2(self.to_numpy(vec)[b, c, :, :])
        
        fft_result = self.to_tensor(fft_result)

        # Colomn wise vectorization
        return fft_result.reshape(vec.shape[0],-1)

    def U(self, vec):
        # Ensure vec is a tensor
        # vec = self.to_tensor(vec)

        # Apply IFFT2 on each channel separately for each batch
        ifft_result = np.zeros_like(self.to_numpy(vec), dtype=np.complex128)

        for b in range(vec.shape[0]):  # Loop over each batch
            for c in range(self.channels):  # Loop over each channel
                # Apply IFFT2 on each 2D channel
                ifft_result[b, c, :, :] = np.fft.ifft2(self.to_numpy(vec)[b, c, :, :])
        
        ifft_result = self.to_tensor(ifft_result)
                # Colomn wise vectorization
        return ifft_result.reshape(vec.shape[0], -1)

    def Ut(self, vec):
        # Ensure vec is a tensor
        # vec = self.to_tensor(vec)

        # Apply IFFT2 on each channel separately for each batch
        fft_result = np.zeros_like(self.to_numpy(vec), dtype=np.complex128)

        for b in range(vec.shape[0]):  # Loop over each batch
            for c in range(self.channels):  # Loop over each channel
                # Apply IFFT2 on each 2D channel
                fft_result[b, c, :, :] = np.fft.fft2(self.to_numpy(vec)[b, c, :, :])
        
        fft_result = self.to_tensor(fft_result)
                # Colomn wise vectorization
        return fft_result.reshape(vec.shape[0], -1)

    def singulars(self):

        # Calculate singular values and return as tensor on the specified device
        Mh, Nh = self.kernel.shape

        # Calculate the center
        center = np.floor(np.array([Mh, Nh]) / 2).astype(int)

        padded_hest = np.pad(self.kernel, ((0, self.dim - Mh), (0, self.dim - Nh)), mode='constant')
        shifted_hest = np.roll(padded_hest, - center, axis=(0, 1))

        # Applying 2D FFT and take the real part
        sing_values = (torch.tensor((np.fft.fft2(shifted_hest)), device='cuda'))
        sing_values = sing_values.unsqueeze(2).expand(-1, -1, self.channels).permute(2, 0, 1) # [C,H,W]

        # import scipy.io
        # scipy.io.savemat('sing.mat', {'sing':torch.tensor((np.fft.fft2(shifted_hest)), device='cuda').T.unsqueeze(0).detach().cpu().numpy(), 'sing_3c': torch.tensor((np.fft.fft2(shifted_hest)), device='cuda').T.unsqueeze(0).repeat(3, 1, 1).detach().cpu().numpy()})

        # colomn wise vectorization
        return sing_values.reshape(-1)

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        S_Vt = (singulars * temp[:, :singulars.shape[0]]).reshape(1,self.channels,self.dim, self.dim)
        output = self.U(S_Vt)
        import scipy.io
        scipy.io.savemat('H_values_bccb.mat',
                         {'Vt': temp.reshape(self.dim, self.dim,self.channels).permute(1,0,2).detach().cpu().numpy(), 'singulars': singulars.detach().cpu().numpy(),
                          'output': output.reshape(self.dim, self.dim,self.channels).permute(1,0,2).detach().cpu().numpy(), 'vec': vec.reshape(self.dim, self.dim,self.channels).detach().cpu().numpy()})
        return output

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        output = self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
        return output
        
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        inv_singulars = torch.where(singulars != 0, 1.0 / singulars, torch.tensor(float('inf')))

        # Replace any infinities with a large value (e.g., 1000)
        inv_singulars = torch.where(torch.isinf(inv_singulars), torch.tensor(1000.0), inv_singulars)

        temp[:, :singulars.shape[0]] = (temp[:, :singulars.shape[0]] * inv_singulars)
        temp = temp.view(1,3,512,512)
        output = self.V(temp)
        import scipy.io
        scipy.io.savemat('H_values.mat',
                         {'Vt': temp.detach().cpu().numpy(), 'singulars': inv_singulars.detach().cpu().numpy(),
                          'output': output.detach().cpu().numpy()})
        
        return self.V(temp)


class deconvolution_BCCB_br(H_functions):


    def __init__(self, kernel, U, S, V, device):
        self.kernel = kernel
        self._U = torch.tensor(U,dtype=torch.float32, device = device)
        self._singulars = torch.tensor(S,dtype=torch.float32, device = device)
        self._V = torch.tensor(V,dtype=torch.float32, device = device)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        self.device = device
        self.channels = 3
        self.dim = 128

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        
        
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,1)).view(v.shape[0], M.shape[0])

    def V(self, vec):
        vec = vec.reshape(1,3,128,128)
        results = []
        for channel in range(self.channels):
            v_channel = vec[:, channel, :, :]  # Shape (N, H, W)
            v_channel_reshaped = v_channel.view(vec.shape[0], -1)  # Shape (N, H*W)
            # Apply mat_by_vec to the reshaped channel
            result_channel = self.mat_by_vec(self._V, v_channel_reshaped)  # Shape (N, M.shape[0])
            results.append(result_channel)

            # Stack the results to maintain the channel dimension
        output = torch.stack(results, dim=1)  # Shape (N, C, M.shape[0])

        return output.reshape(vec.shape[0], -1)  # Return the output in the shape (N, C * M.shape[0])

        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        vec = vec.reshape(1,3,128,128)
        results = []
        for channel in range(self.channels):
            v_channel = vec[:, channel, :, :]  # Shape (N, H, W)
            v_channel_reshaped = v_channel.view(vec.shape[0], -1)  # Shape (N, H*W)
            # Apply mat_by_vec to the reshaped channel
            result_channel = self.mat_by_vec(self._Vt, v_channel_reshaped)  # Shape (N, M.shape[0])
            results.append(result_channel)

            # Stack the results to maintain the channel dimension
        output = torch.stack(results, dim=1)  # Shape (N, C, M.shape[0])

        return output.reshape(vec.shape[0], -1)  # Return the output in the shape (N, C * M.shape[0])

    def U(self, vec):
        vec = vec.reshape(1,3,128,128)
        results = []
        for channel in range(self.channels):
            v_channel = vec[:, channel, :, :]  # Shape (N, H, W)
            v_channel_reshaped = v_channel.view(vec.shape[0], -1)  # Shape (N, H*W)
            # Apply mat_by_vec to the reshaped channel
            result_channel = self.mat_by_vec(self._U, v_channel_reshaped)  # Shape (N, M.shape[0])
            results.append(result_channel)

            # Stack the results to maintain the channel dimension
        output = torch.stack(results, dim=1)  # Shape (N, C, M.shape[0])

        return output.reshape(vec.shape[0], -1)  # Return the output in the shape (N, C * M.shape[0])

    def Ut(self, vec):
        vec = vec.reshape(1,3,128,128)
        results = []
        for channel in range(self.channels):
            v_channel = vec[:, channel, :, :]  # Shape (N, H, W)
            v_channel_reshaped = v_channel.view(vec.shape[0], -1)  # Shape (N, H*W)
            # Apply mat_by_vec to the reshaped channel
            result_channel = self.mat_by_vec(self._Ut, v_channel_reshaped)  # Shape (N, M.shape[0])
            results.append(result_channel)

            # Stack the results to maintain the channel dimension
        output = torch.stack(results, dim=1)  # Shape (N, C, M.shape[0])

        return output.reshape(vec.shape[0], -1)  # Return the output in the shape (N, C * M.shape[0])

    def singulars(self):
        return torch.diag(self._singulars).repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio ** 2] * ratio ** 2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim ** 2, self.ratio ** 2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim ** 2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio ** 2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1].view(
                vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                              self.channels, -1,
                                                                                              self.ratio ** 2)
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                               self.channels, -1,
                                                                                               self.ratio ** 2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        recon[:, :self.channels * self.y_dim ** 2] = patches[:, :, :, 0].view(vec.shape[0],
                                                                              self.channels * self.y_dim ** 2)
        for idx in range(self.ratio ** 2 - 1):
            recon[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_dim ** 2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


class SuperResolution_2(H_functions):

    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, ratio, device):
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        # build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0] // 2, i + kernel1.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0] // 2]
        # build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0] // 2, i + kernel2.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1),
                                       self.singulars_small2.reshape(1, img_dim)).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim ** 2, self.ratio ** 2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim ** 2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio ** 2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1].view(
                vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                              self.channels, -1,
                                                                                              self.ratio ** 2)
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                               self.channels, -1,
                                                                                               self.ratio ** 2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        recon[:, :self.channels * self.y_dim ** 2] = patches[:, :, :, 0].view(vec.shape[0],
                                                                              self.channels * self.y_dim ** 2)
        for idx in range(self.ratio ** 2 - 1):
            recon[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_dim ** 2)
        return recon

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        # Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,
                                                                                            self.channels)  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,
                                                                                             self.channels)  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim ** 2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        temp[:, :self.img_dim ** 2] = reshaped
        return temp


# Walsh-Hadamard Compressive Sensing
class WalshHadamardCS(H_functions):
    def fwht(self, vec):  # the Fast Walsh Hadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim ** 2)
        h = 1
        while h < self.img_dim ** 2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2 * h]
            a[:, :, :, h:2 * h] = b[:, :, :, :h] - b[:, :, :, h:2 * h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim ** 2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim ** 2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device)
        out[:, :self.channels * self.img_dim ** 2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                                         dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                                      dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        # build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # reflective padding
                if j_effective < 0: j_effective = -j_effective - 1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                # matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1),
                                       self.singulars_small.reshape(1, small_dim)).reshape(small_dim ** 2)
        # permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in
                                   range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)[:,
                                 :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)[:,
                                           self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        # permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim ** 2, self.channels, device=vec.device)
        temp[:, :self.small_dim ** 2, :] = vec.clone().reshape(vec.shape[0], self.small_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        # permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


# Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO=3e-2):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1),
                                       self.singulars_small.reshape(1, img_dim)).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        
        import scipy.io

        scipy.io.savemat('data.mat', {'vec': vec.detach().cpu().numpy(), 'output': temp.detach().cpu().numpy(),
                                      'vec_output': temp.reshape(vec.shape[0], -1).reshape(vec.shape[0], -1).detach().cpu().numpy()})
       
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        import scipy.io
        scipy.io.savemat('H_values.mat', {'singulars_vec': self._singulars.repeat(1, 3).reshape(-1).detach().cpu().numpy(),'singulars_vec': self._singulars.repeat(1, 3).reshape(-1).detach().cpu().numpy()})
    
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                                         self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                                      self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0] // 2, i + kernel1.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0] // 2]
        # build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0] // 2, i + kernel2.shape[0] // 2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1),
                                       self.singulars_small2.reshape(1, img_dim)).reshape(img_dim ** 2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True)  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)