"""
Model loader utilities for LLMdoctor framework.

Provides unified interface for loading and managing PatientModel and DoctorModel instances.
"""

import torch
from typing import Dict, Optional, Union, Tuple
import logging
from pathlib import Path

from .patient_model import PatientModel
from .doctor_model import DoctorModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Unified model loader for LLMdoctor framework.
    
    Handles loading of both PatientModel and DoctorModel with proper
    device management and memory optimization.
    """
    
    @staticmethod
    def load_patient_model(
        model_name_or_path: str,
        device: Union[str, torch.device] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> PatientModel:
        """
        Load a PatientModel instance.
        
        Args:
            model_name_or_path: Model path or HuggingFace model name
            device: Device to load on
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            **kwargs: Additional arguments for PatientModel
            
        Returns:
            Loaded PatientModel instance
        """
        logger.info(f"Loading PatientModel: {model_name_or_path}")
        
        patient_model = PatientModel(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            **kwargs
        )
        
        logger.info("PatientModel loaded successfully")
        return patient_model
    
    @staticmethod
    def load_doctor_model(
        model_name_or_path: str,
        num_preference_dims: int = 1,
        device: Union[str, torch.device] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        freeze_base_model: bool = False,
        **kwargs
    ) -> DoctorModel:
        """
        Load a DoctorModel instance.
        
        Args:
            model_name_or_path: Model path or HuggingFace model name
            num_preference_dims: Number of preference dimensions
            device: Device to load on
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            freeze_base_model: Whether to freeze base model parameters
            **kwargs: Additional arguments for DoctorModel
            
        Returns:
            Loaded DoctorModel instance
        """
        logger.info(f"Loading DoctorModel: {model_name_or_path}")
        
        # Check if it's a pre-trained LLMdoctor model
        if Path(model_name_or_path).exists() and (Path(model_name_or_path) / "value_heads.pt").exists():
            doctor_model = DoctorModel.load_pretrained(
                load_path=model_name_or_path,
                device=device,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                freeze_base_model=freeze_base_model,
                **kwargs
            )
        else:
            doctor_model = DoctorModel(
                model_name_or_path=model_name_or_path,
                num_preference_dims=num_preference_dims,
                device=device,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                freeze_base_model=freeze_base_model,
                **kwargs
            )
        
        logger.info("DoctorModel loaded successfully")
        return doctor_model
    
    @staticmethod
    def load_model_pair(
        patient_model_path: str,
        doctor_model_path: str,
        num_preference_dims: int = 1,
        device: Union[str, torch.device] = "auto",
        patient_dtype: torch.dtype = torch.float16,
        doctor_dtype: torch.dtype = torch.float16,
        patient_8bit: bool = False,
        patient_4bit: bool = False,
        doctor_8bit: bool = False,
        doctor_4bit: bool = False,
        **kwargs
    ) -> Tuple[PatientModel, DoctorModel]:
        """
        Load a patient-doctor model pair.
        
        Args:
            patient_model_path: Path to patient model
            doctor_model_path: Path to doctor model
            num_preference_dims: Number of preference dimensions for doctor
            device: Device to load models on
            patient_dtype: Data type for patient model
            doctor_dtype: Data type for doctor model
            patient_8bit: Whether to use 8-bit quantization for patient
            patient_4bit: Whether to use 4-bit quantization for patient
            doctor_8bit: Whether to use 8-bit quantization for doctor
            doctor_4bit: Whether to use 4-bit quantization for doctor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (PatientModel, DoctorModel)
        """
        logger.info("Loading patient-doctor model pair")
        
        # Load patient model
        patient_model = ModelLoader.load_patient_model(
            model_name_or_path=patient_model_path,
            device=device,
            torch_dtype=patient_dtype,
            load_in_8bit=patient_8bit,
            load_in_4bit=patient_4bit,
            **kwargs
        )
        
        # Load doctor model
        doctor_model = ModelLoader.load_doctor_model(
            model_name_or_path=doctor_model_path,
            num_preference_dims=num_preference_dims,
            device=device,
            torch_dtype=doctor_dtype,
            load_in_8bit=doctor_8bit,
            load_in_4bit=doctor_4bit,
            **kwargs
        )
        
        logger.info("Model pair loaded successfully")
        return patient_model, doctor_model
    
    @staticmethod
    def get_optimal_device_config(
        model_size: str = "7b",
        available_memory_gb: Optional[float] = None
    ) -> Dict[str, Union[str, bool, torch.dtype]]:
        """
        Get optimal device configuration for model loading.
        
        Args:
            model_size: Model size indicator ("7b", "13b", "70b")
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Dictionary with optimal configuration
        """
        config = {
            "device": "auto",
            "torch_dtype": torch.float16,
            "load_in_8bit": False,
            "load_in_4bit": False,
        }
        
        if not torch.cuda.is_available():
            config["device"] = "cpu"
            config["torch_dtype"] = torch.float32
            return config
        
        # Get available GPU memory
        if available_memory_gb is None:
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Rough memory requirements (in GB) for different model sizes
        memory_requirements = {
            "7b": {"fp16": 14, "8bit": 7, "4bit": 4},
            "13b": {"fp16": 26, "8bit": 13, "4bit": 7},
            "70b": {"fp16": 140, "8bit": 70, "4bit": 35},
        }
        
        size_key = model_size.lower().replace("b", "b")
        if size_key not in memory_requirements:
            size_key = "7b"  # Default fallback
        
        reqs = memory_requirements[size_key]
        
        # Choose quantization based on available memory
        if available_memory_gb >= reqs["fp16"]:
            # Enough memory for full precision
            pass
        elif available_memory_gb >= reqs["8bit"]:
            # Use 8-bit quantization
            config["load_in_8bit"] = True
        elif available_memory_gb >= reqs["4bit"]:
            # Use 4-bit quantization
            config["load_in_4bit"] = True
        else:
            # Fallback to CPU
            config["device"] = "cpu"
            config["torch_dtype"] = torch.float32
            logger.warning(f"Insufficient GPU memory ({available_memory_gb:.1f}GB), falling back to CPU")
        
        return config
    
    @staticmethod
    def estimate_memory_usage(
        model_size: str = "7b",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ) -> float:
        """
        Estimate memory usage for model loading.
        
        Args:
            model_size: Model size indicator
            torch_dtype: Data type
            load_in_8bit: Whether using 8-bit quantization
            load_in_4bit: Whether using 4-bit quantization
            
        Returns:
            Estimated memory usage in GB
        """
        # Base parameter counts (approximate)
        param_counts = {
            "7b": 7e9,
            "13b": 13e9,
            "70b": 70e9,
        }
        
        size_key = model_size.lower().replace("b", "b")
        if size_key not in param_counts:
            size_key = "7b"
        
        param_count = param_counts[size_key]
        
        # Calculate memory based on quantization
        if load_in_4bit:
            bytes_per_param = 0.5  # 4-bit
        elif load_in_8bit:
            bytes_per_param = 1.0  # 8-bit
        elif torch_dtype == torch.float16:
            bytes_per_param = 2.0  # 16-bit
        else:
            bytes_per_param = 4.0  # 32-bit
        
        # Add overhead for activations, gradients, etc.
        base_memory_gb = (param_count * bytes_per_param) / 1024**3
        total_memory_gb = base_memory_gb * 1.5  # 50% overhead
        
        return total_memory_gb
    
    @staticmethod
    def validate_model_compatibility(
        patient_model: PatientModel,
        doctor_model: DoctorModel
    ) -> bool:
        """
        Validate that patient and doctor models are compatible.
        
        Args:
            patient_model: Patient model instance
            doctor_model: Doctor model instance
            
        Returns:
            True if models are compatible
        """
        try:
            # Check tokenizer compatibility
            patient_vocab_size = patient_model.vocab_size
            doctor_vocab_size = doctor_model.vocab_size
            
            if patient_vocab_size != doctor_vocab_size:
                logger.warning(
                    f"Vocabulary size mismatch: Patient={patient_vocab_size}, "
                    f"Doctor={doctor_vocab_size}"
                )
                return False
            
            # Check tokenizer special tokens
            patient_tokens = {
                "pad": patient_model.tokenizer.pad_token_id,
                "eos": patient_model.tokenizer.eos_token_id,
                "bos": getattr(patient_model.tokenizer, "bos_token_id", None),
            }
            
            doctor_tokens = {
                "pad": doctor_model.tokenizer.pad_token_id,
                "eos": doctor_model.tokenizer.eos_token_id,
                "bos": getattr(doctor_model.tokenizer, "bos_token_id", None),
            }
            
            if patient_tokens != doctor_tokens:
                logger.warning(f"Special token mismatch: Patient={patient_tokens}, Doctor={doctor_tokens}")
                return False
            
            logger.info("Model compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during model compatibility validation: {e}")
            return False