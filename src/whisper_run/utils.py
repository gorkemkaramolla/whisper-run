import time
from typing import Callable, Any, Tuple

def measure_time(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time
