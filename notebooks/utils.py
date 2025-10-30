import tiktoken

def calculate_context_percentage(
    text: str,
    model: str = 'gpt-4o',
    context_window: int = 128000
) -> dict:
    """
    Calculate what percentage of context a text uses
    
    Args:
        text: The text string to analyze
        model: Model name (default: 'gpt-4o')
        context_window: Context window size (default: 128000)
    
    Returns:
        dict with tokens, percentage, and other stats
    
    Example:
        >>> result = calculate_context_percentage("Hello world", "gpt-4o")
        >>> print(f"Uses {result['percentage']:.2f}% of context")
    """
    
    # Get tokenizer
    try:
        if model.startswith('gpt-4') or model.startswith('o'):
            encoding = tiktoken.get_encoding('cl100k_base')
        else:
            encoding = tiktoken.get_encoding('cl100k_base')  # Default
        
        tokens = len(encoding.encode(text))
    except:
        # Fallback: estimate (1 word ≈ 1.3 tokens)
        tokens = int(len(text.split()) * 1.3)
    
    # Calculate percentage
    percentage = (tokens / context_window) * 100
    
    return {
        'tokens': tokens,
        'percentage': percentage,
        'context_window': context_window,
        'remaining': context_window - tokens,
        'fits': tokens < context_window
    }
