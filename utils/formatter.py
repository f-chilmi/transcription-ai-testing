import json

def format_transcript(json_data):
    """
    Format WhisperX transcript JSON into speaker: text format
    
    Args:
        json_data: Dictionary containing transcript data or JSON string
    
    Returns:
        Formatted string with speaker labels and text
    """
    # Handle both dict and JSON string input
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract segments
    segments = data.get('segments', {}).get('segments', [])
    
    # Format output
    formatted_lines = []
    
    for segment in segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment.get('text', '').strip()
        
        if text:  # Only add non-empty text
            formatted_lines.append(f"{speaker}: {text}")
    
    return '\n'.join(formatted_lines)

def format_transcript_with_timestamps(json_data):
    """
    Format transcript with timestamps included
    
    Args:
        json_data: Dictionary containing transcript data or JSON string
    
    Returns:
        Formatted string with timestamps, speaker labels and text
    """
    # Handle both dict and JSON string input
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract segments
    segments = data.get('segments', {}).get('segments', [])
    
    # Format output
    formatted_lines = []
    
    for segment in segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment.get('text', '').strip()
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        
        if text:  # Only add non-empty text
            # Format timestamps as MM:SS
            start_min, start_sec = divmod(int(start_time), 60)
            end_min, end_sec = divmod(int(end_time), 60)
            
            timestamp = f"[{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}]"
            formatted_lines.append(f"{timestamp} {speaker}: {text}")
    
    return '\n'.join(formatted_lines)

# Example usage with your data
if __name__ == "__main__":
    
    # print("=== Basic Format ===")
    # print(format_transcript(sample_data))
    
    # print("\n=== With Timestamps ===")
    # print(format_transcript_with_timestamps(sample_data))
    
    # If you want to read from a JSON file:
    with open('audio_mono_arabic_whisperx_transcription_diarization.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        formatted = format_transcript(data)
        print(formatted)
    
    # If you want to save the formatted output:
    # with open('formatted_transcript.txt', 'w', encoding='utf-8') as f:
    #     f.write(format_transcript(sample_data))