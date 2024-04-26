import torch
import numpy as np
import json
import gc
import logging
# Create a custom logger
logger = logging.getLogger("LaserRMTrainer | Scanning")

# Set level of logging
logger.setLevel(logging.DEBUG)  # Set to lowest level needed

# Create handlers
c_handler = logging.StreamHandler()  # This outputs to sys.stdout
f_handler = logging.FileHandler("file.log")

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

class ModelModifier:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model  # AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}

    def calculate_snr_for_layer(self, layer_type, layer_number):
        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                weights = module.weight.double()
                S = torch.linalg.svdvals(weights)
                max_singular_value = S[0].item()  # First singularity value
                weights = weights.detach().cpu()
                S = S.detach().cpu()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)

                signal = S[S > mp_threshold].sum()
                noise = S[S <= mp_threshold].sum()
                snr = signal / noise if noise != 0 else float('inf')
                snr_ratio = snr / max_singular_value  # Calculates the ratio of SNR to the highest singularity value
                del S, weights
                torch.cuda.empty_cache()  # Clear PyTorch's CUDA memory cache
                gc.collect()
                return snr_ratio  # Returns the ratio
    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold

    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349 ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated


    def assess_layers_snr(self, layer_types, layer_numbers):
        for name, module in self.model.named_modules():
            for layer_number in layer_numbers:
                for layer_type in layer_types:
                    if layer_type in name and str(layer_number) in name:
                        print("*"*50, flush=True)
                        print(f"Calculating Signal to Noise Ratio at layer {name}", flush=True)
                        snr_ratio = self.calculate_snr_for_layer(layer_type, layer_number)
                        self.layer_snr[name] = {'snr_ratio': snr_ratio, 'module': name}
                        print(f"Signal to Noise Ratio at layer {name} = {snr_ratio}", flush=True)
                        print("*"*50, flush=True)


    def save_layers_to_json(self, filename="layer_snr_info.json"):
        with open(filename, 'w') as file:
            serializable_data = {}
            for key, value in self.layer_snr.items():
                # Convert Tensors to Python numbers (for SNR) and handle other data types as needed
                snr_value = value['snr_ratio'].item() if isinstance(value['snr_ratio'], torch.Tensor) else value['snr_ratio']
                module_str = str(value['module'])  # Assuming module representation is a string or convertible to a string
                serializable_data[key] = {'snr': snr_value, 'module': module_str}

            json.dump(serializable_data, file, indent=4)

    def get_bottom_snr_ratios(self, bottom_n=16):
        # Initialize a dictionary to store the SNR ratios for the specific modules
        snr_ratios_per_specific_module = {
            'self_attn.v_proj': [],
            'self_attn.k_proj': [],
            'self_attn.o_proj': [],
            'self_attn.q_proj': [],
            'mlp.down_proj': [],
            'mlp.up_proj': [],
            'mlp.gate_proj': []
        }

        # Run through all layer SNR entries
        for name, value in self.layer_snr.items():
            snr_ratio = value['snr_ratio']
            layer_name = value['module']

            # For each specific module, check if the layer name contains the module
            for specific_module in snr_ratios_per_specific_module.keys():
                if specific_module in layer_name:
                    # Add the layer name and SNR value to the corresponding entry
                    snr_ratios_per_specific_module[specific_module].append((layer_name, snr_ratio))
                    break  # End the loop when the module is found to avoid duplicate entries

        # Sort and extract the bottom 16 SNR values for each specific module
        bottom_snr_layers = {}
        for module, snr_ratios in snr_ratios_per_specific_module.items():
            sorted_layers = sorted(snr_ratios, key=lambda x: x[1])  # Sort by SNR value
            bottom_snr_layers[module] = [layer[0] for layer in sorted_layers[:bottom_n]]  # Saving the layer names

        return bottom_snr_layers

    def get_top_snr_ratios(self, top_n=16):
        # Initialize a dictionary to store the SNR ratios for the specific modules
        snr_ratios_per_specific_module = {
            'self_attn.v_proj': [],
            'self_attn.k_proj': [],
            'self_attn.o_proj': [],
            'self_attn.q_proj': [],
            'mlp.down_proj': [],
            'mlp.up_proj': [],
            'mlp.gate_proj': []
        }

        # Run through all layer SNR entries
        for name, value in self.layer_snr.items():
            snr_ratio = value['snr_ratio']
            layer_name = value['module']

            # For each specific module, check if the layer name contains the module
            for specific_module in snr_ratios_per_specific_module.keys():
                if specific_module in layer_name:
                    # Add the layer name and SNR value to the corresponding entry
                    snr_ratios_per_specific_module[specific_module].append((layer_name, snr_ratio))
                    break  # End the loop when the module is found to avoid duplicate entries

        # Sort and extract the top 16 SNR values for each specific module
        top_snr_layers = {}
        for module, snr_ratios in snr_ratios_per_specific_module.items():
            sorted_layers = sorted(snr_ratios, key=lambda x: x[1], reverse=True)  # Sort by SNR value
            top_snr_layers[module] = [layer[0] for layer in sorted_layers[:top_n]]  # Saving the layer names

        return top_snr_layers


    def save_top_snr_ratios_to_json(self, top_snr_layers, filename="top_snr_ratios.json"):
        with open(filename, 'w') as file:
            json.dump(top_snr_layers, file, indent=4)

    def save_bottom_snr_ratios_to_json(self, bottom_snr_layers, filename="bottom_snr_ratios.json"):
        with open(filename, 'w') as file:
            json.dump(bottom_snr_layers, file, indent=4)