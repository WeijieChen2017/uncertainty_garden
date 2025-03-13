import numpy as np
from typing import Union, Callable, Dict, List, Optional, Tuple
from scipy import stats


def preprocess_data(observations: np.ndarray, 
                    clip_min: Optional[float] = None, 
                    clip_max: Optional[float] = None,
                    ensure_non_negative: bool = False) -> np.ndarray:
    """
    Preprocess input data for uncertainty quantification.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        ensure_non_negative: If True, shift data to ensure all values are non-negative
        
    Returns:
        Preprocessed observations array
    """
    processed_data = observations.copy()
    
    # Clip data if specified
    if clip_min is not None or clip_max is not None:
        processed_data = np.clip(processed_data, clip_min, clip_max)
    
    # Ensure non-negative values if required
    if ensure_non_negative:
        min_val = np.min(processed_data)
        if min_val < 0:
            processed_data = processed_data - min_val
    
    return processed_data


def variance(observations: np.ndarray, voxel_wise: bool = True, 
             clip_min: Optional[float] = None, 
             clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate variance across multiple observations.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
    
    Returns:
        Variance as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    # Calculate variance along the first axis (across observations)
    var = np.var(processed_data, axis=0)
    
    if voxel_wise:
        return var
    else:
        # Return average variance across all voxels
        return np.mean(var)


def standard_deviation(observations: np.ndarray, voxel_wise: bool = True,
                       clip_min: Optional[float] = None, 
                       clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate standard deviation across multiple observations.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
    
    Returns:
        Standard deviation as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    # Calculate standard deviation along the first axis (across observations)
    std = np.std(processed_data, axis=0)
    
    if voxel_wise:
        return std
    else:
        # Return average standard deviation across all voxels
        return np.mean(std)


def coefficient_of_variation(observations: np.ndarray, voxel_wise: bool = True, 
                             epsilon: float = 1e-10,
                             clip_min: Optional[float] = None, 
                             clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate coefficient of variation (CV) across multiple observations.
    CV = standard deviation / mean
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        epsilon: Small value to avoid division by zero
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Coefficient of variation as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data - ensure non-negative for CV
    processed_data = preprocess_data(observations, clip_min, clip_max, ensure_non_negative=True)
    
    mean = np.mean(processed_data, axis=0)
    std = np.std(processed_data, axis=0)
    
    # Avoid division by zero
    cv = std / (mean + epsilon)
    
    if voxel_wise:
        return cv
    else:
        # Return average CV across all voxels
        return np.mean(cv)


def entropy(observations: np.ndarray, voxel_wise: bool = True, 
            bins: int = 10,
            clip_min: Optional[float] = None, 
            clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate entropy of the distribution across multiple observations.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        bins: Number of bins for histogram calculation
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Entropy as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data - ensure non-negative for entropy calculation
    processed_data = preprocess_data(observations, clip_min, clip_max, ensure_non_negative=True)
    
    N, x, y, z = processed_data.shape
    
    if voxel_wise:
        entropy_map = np.zeros((x, y, z))
        
        # Calculate entropy for each voxel position
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # Get all observations for this voxel
                    voxel_values = processed_data[:, i, j, k]
                    
                    # Calculate histogram
                    hist, _ = np.histogram(voxel_values, bins=bins, density=True)
                    
                    # Remove zeros to avoid log(0)
                    hist = hist[hist > 0]
                    
                    # Calculate entropy
                    entropy_map[i, j, k] = -np.sum(hist * np.log2(hist))
        
        return entropy_map
    else:
        # Calculate average entropy across all voxels
        total_entropy = 0
        count = 0
        
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    voxel_values = processed_data[:, i, j, k]
                    hist, _ = np.histogram(voxel_values, bins=bins, density=True)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        total_entropy += -np.sum(hist * np.log2(hist))
                        count += 1
        
        return total_entropy / count if count > 0 else 0


def interquartile_range(observations: np.ndarray, voxel_wise: bool = True,
                        clip_min: Optional[float] = None, 
                        clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate interquartile range (IQR) across multiple observations.
    IQR = Q3 - Q1 (75th percentile - 25th percentile)
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        IQR as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    q1 = np.percentile(processed_data, 25, axis=0)
    q3 = np.percentile(processed_data, 75, axis=0)
    iqr = q3 - q1
    
    if voxel_wise:
        return iqr
    else:
        # Return average IQR across all voxels
        return np.mean(iqr)


def range_width(observations: np.ndarray, voxel_wise: bool = True,
                clip_min: Optional[float] = None, 
                clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate range (max - min) across multiple observations.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Range as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    max_vals = np.max(processed_data, axis=0)
    min_vals = np.min(processed_data, axis=0)
    range_vals = max_vals - min_vals
    
    if voxel_wise:
        return range_vals
    else:
        # Return average range across all voxels
        return np.mean(range_vals)


def confidence_interval_width(observations: np.ndarray, voxel_wise: bool = True, 
                              confidence: float = 0.95,
                              clip_min: Optional[float] = None, 
                              clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate width of confidence interval across multiple observations.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        confidence: Confidence level (default: 0.95 for 95% confidence interval)
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Confidence interval width as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    N = processed_data.shape[0]
    
    # Calculate standard error of the mean
    sem = np.std(processed_data, axis=0) / np.sqrt(N)
    
    # Calculate t-value for the given confidence level
    t_value = stats.t.ppf((1 + confidence) / 2, N - 1)
    
    # Calculate confidence interval width
    ci_width = 2 * t_value * sem
    
    if voxel_wise:
        return ci_width
    else:
        # Return average CI width across all voxels
        return np.mean(ci_width)


def predictive_variance(observations: np.ndarray, voxel_wise: bool = True,
                        clip_min: Optional[float] = None, 
                        clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate predictive variance for ensemble models.
    For Bayesian models or ensembles, this combines both aleatoric and epistemic uncertainty.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Predictive variance as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data
    processed_data = preprocess_data(observations, clip_min, clip_max)
    
    # Calculate mean prediction
    mean_prediction = np.mean(processed_data, axis=0)
    
    # Calculate predictive variance
    pred_var = np.mean(np.square(processed_data - mean_prediction[np.newaxis, ...]), axis=0)
    
    if voxel_wise:
        return pred_var
    else:
        # Return average predictive variance across all voxels
        return np.mean(pred_var)


def mutual_information(observations: np.ndarray, voxel_wise: bool = True, 
                       bins: int = 10,
                       clip_min: Optional[float] = None, 
                       clip_max: Optional[float] = None) -> Union[np.ndarray, float]:
    """
    Calculate mutual information for Bayesian models.
    This is a measure of epistemic uncertainty.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        bins: Number of bins for histogram calculation
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Mutual information as either voxel-wise array (x, y, z) or a single average value
    """
    # Preprocess data - ensure non-negative for entropy calculation
    processed_data = preprocess_data(observations, clip_min, clip_max, ensure_non_negative=True)
    
    N, x, y, z = processed_data.shape
    
    # Initialize mutual information map
    if voxel_wise:
        mi_map = np.zeros((x, y, z))
        
        # Calculate mutual information for each voxel
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # Get all observations for this voxel
                    voxel_values = processed_data[:, i, j, k]
                    
                    # Calculate entropy of the predictive distribution
                    hist, _ = np.histogram(voxel_values, bins=bins, density=True)
                    hist = hist[hist > 0]  # Remove zeros
                    if len(hist) > 0:
                        entropy_pred = -np.sum(hist * np.log2(hist))
                    else:
                        entropy_pred = 0
                    
                    # Calculate average entropy of individual predictions
                    # (simplified approach for demonstration)
                    avg_entropy = 0
                    
                    # Mutual information = entropy_pred - avg_entropy
                    mi_map[i, j, k] = entropy_pred - avg_entropy
        
        return mi_map
    else:
        # Calculate average mutual information across all voxels
        total_mi = 0
        count = 0
        
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    voxel_values = processed_data[:, i, j, k]
                    hist, _ = np.histogram(voxel_values, bins=bins, density=True)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        entropy_pred = -np.sum(hist * np.log2(hist))
                        total_mi += entropy_pred  # Simplified MI calculation
                        count += 1
        
        return total_mi / count if count > 0 else 0


def get_uncertainty_metric(metric_name: str) -> Callable:
    """
    Get uncertainty metric function by name.
    
    Args:
        metric_name: Name of the uncertainty metric
        
    Returns:
        Function that calculates the specified uncertainty metric
    """
    metrics = {
        'variance': variance,
        'std': standard_deviation,
        'cv': coefficient_of_variation,
        'entropy': entropy,
        'iqr': interquartile_range,
        'range': range_width,
        'ci_width': confidence_interval_width,
        'pred_var': predictive_variance,
        'mutual_info': mutual_information
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Unknown uncertainty metric: {metric_name}. "
                         f"Available metrics: {list(metrics.keys())}")
    
    return metrics[metric_name]


def calculate_all_metrics(observations: np.ndarray, voxel_wise: bool = True,
                          clip_min: Optional[float] = None, 
                          clip_max: Optional[float] = None) -> Dict:
    """
    Calculate all uncertainty metrics.
    
    Args:
        observations: Input array of shape (N, x, y, z) where N is the number of observations
        voxel_wise: If True, return voxel-wise uncertainty of shape (x, y, z),
                    otherwise return average uncertainty as a single float
        clip_min: Minimum value for clipping (None for no minimum clipping)
        clip_max: Maximum value for clipping (None for no maximum clipping)
        
    Returns:
        Dictionary containing all uncertainty metrics
    """
    metrics = {
        'variance': variance(observations, voxel_wise, clip_min, clip_max),
        'std': standard_deviation(observations, voxel_wise, clip_min, clip_max),
        'cv': coefficient_of_variation(observations, voxel_wise, clip_min=clip_min, clip_max=clip_max),
        'entropy': entropy(observations, voxel_wise, clip_min=clip_min, clip_max=clip_max),
        'iqr': interquartile_range(observations, voxel_wise, clip_min, clip_max),
        'range': range_width(observations, voxel_wise, clip_min, clip_max),
        'ci_width': confidence_interval_width(observations, voxel_wise, clip_min=clip_min, clip_max=clip_max),
        'pred_var': predictive_variance(observations, voxel_wise, clip_min, clip_max),
        'mutual_info': mutual_information(observations, voxel_wise, clip_min=clip_min, clip_max=clip_max)
    }
    
    return metrics
