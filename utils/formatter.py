import json
from typing import Dict, List, Union

def extract_speaker_text_simple(json_data: Union[str, dict]) -> Dict[str, List[str]]:
    """
    Extract speaker text from JSON, grouping all text by speaker.
    
    Args:
        json_data: JSON string or already parsed dict
        
    Returns:
        Dict with speaker names as keys and list of their texts as values
        Example: {"SPEAKER_00": ["Hello", "How are you?"], "SPEAKER_01": ["Hi there"]}
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    speaker_texts = {}
    
    # Navigate through the nested structure
    try:
        # Get segments from the nested structure
        segments = data["results"]["mono"]["silero_vad"]["result"]["segments"]
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            
            if text:  # Only add non-empty text
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = []
                speaker_texts[speaker].append(text)
    
    except KeyError as e:
        print(f"Key not found in JSON structure: {e}")
        return {}
    
    return speaker_texts

def extract_speaker_text_combined(json_data: Union[str, dict]) -> Dict[str, str]:
    """
    Extract speaker text from JSON, combining all text per speaker into single string.
    
    Args:
        json_data: JSON string or already parsed dict
        
    Returns:
        Dict with speaker names as keys and combined text as values
        Example: {"SPEAKER_00": "Hello How are you?", "SPEAKER_01": "Hi there"}
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    speaker_texts = {}
    
    try:
        segments = data["results"]["mono"]["silero_vad"]["result"]["segments"]
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            
            if text:
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = ""
                else:
                    speaker_texts[speaker] += " "
                speaker_texts[speaker] += text
    
    except KeyError as e:
        print(f"Key not found in JSON structure: {e}")
        return {}
    
    return speaker_texts

def extract_speaker_text_with_timestamps(json_data: Union[str, dict]) -> Dict[str, List[Dict]]:
    """
    Extract speaker text with timestamps preserved.
    
    Args:
        json_data: JSON string or already parsed dict
        
    Returns:
        Dict with speaker names and their segments with timestamps
        Example: {"SPEAKER_00": [{"start": 0.0, "end": 2.5, "text": "Hello"}]}
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    speaker_segments = {}
    
    try:
        segments = data["results"]["mono"]["silero_vad"]["result"]["segments"]
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            
            speaker_segments[speaker].append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip()
            })
    
    except KeyError as e:
        print(f"Key not found in JSON structure: {e}")
        return {}
    
    return speaker_segments

def print_speaker_text_formatted(json_data: Union[str, dict]):
    """
    Print speaker text in a nice formatted way.
    
    Args:
        json_data: JSON string or already parsed dict
    """
    speaker_texts = extract_speaker_text_combined(json_data)
    
    print("=" * 50)
    print("SPEAKER TRANSCRIPTION")
    print("=" * 50)
    
    for speaker, text in speaker_texts.items():
        print(f"\n{speaker}:")
        print("-" * len(speaker))
        print(text)
    
    print("\n" + "=" * 50)

def extract_conversation_flow(json_data: Union[str, dict]) -> List[Dict]:
    """
    Extract conversation in chronological order.
    
    Args:
        json_data: JSON string or already parsed dict
        
    Returns:
        List of segments in order with speaker and text
        Example: [{"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0}]
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    conversation = []
    
    try:
        segments = data["whisperx_diarization"]["segments"]
        # segments = data["results"]["mono"]["silero_vad"]["result"]["segments"]
        
        # Sort by start time to maintain chronological order
        sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))
        
        for segment in sorted_segments:
            conversation.append({
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": segment.get("text", "").strip(),
                "start": segment.get("start"),
                "end": segment.get("end")
            })
    
    except KeyError as e:
        print(f"Key not found in JSON structure: {e}")
        return []
    
    return conversation

# Example usage functions
def load_and_extract_from_file(filename: str) -> Dict[str, str]:
    """
    Load JSON file and extract speaker text.
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Dict with speaker: text mapping
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Test the functions
        # print("1. Simple extraction (list per speaker):")
        # result1 = extract_speaker_text_simple(data)
        # print(result1)
        
        # print("\n2. Combined text per speaker:")
        # result2 = extract_speaker_text_combined(data)
        # print(result2)
        
        # print("\n3. Formatted output:")
        # print_speaker_text_formatted(data)
        
        print("\n4. Conversation flow:")
        result4 = extract_conversation_flow(data)
        for item in result4:
            print(f"[{item['start']:.1f}s] {item['speaker']}: {item['text']}")
        
        # To use with your file:
        # speaker_texts = load_and_extract_from_file("your_file.json")
        # print(speaker_texts)

        # return extract_speaker_text_combined(data)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in file {filename}: {e}")
        return {}

# Example usage:
if __name__ == "__main__":
    json_file = "../transcription_test_results_25-06-2025 17:17:40_vad.json"
    # Example JSON data (you would load this from your file)
    sample_json = {
        "test_start_time": "2025-06-25T17:17:41.569348",
        "results": {
            "mono": {
                "silero_vad": {
                    "result": {
                        "segments": [
                            {"start": 0.0, "end": 2.5, "text": "Hello there", "speaker": "SPEAKER_00"},
                            {"start": 2.6, "end": 4.1, "text": "Hi, how are you?", "speaker": "SPEAKER_01"},
                            {"start": 4.2, "end": 6.0, "text": "I'm doing well", "speaker": "SPEAKER_00"},
                            {"start": 6.1, "end": 8.0, "text": "That's great to hear", "speaker": "SPEAKER_01"}
                        ]
                    }
                }
            }
        }
    }
    
    # Test the functions
    # print("1. Simple extraction (list per speaker):")
    # result1 = extract_speaker_text_simple(json_file)
    # print(result1)
    
    # print("\n2. Combined text per speaker:")
    # result2 = extract_speaker_text_combined(json_file)
    # print(result2)
    
    # print("\n3. Formatted output:")
    # print_speaker_text_formatted(json_file)
    
    # print("\n4. Conversation flow:")
    # result4 = extract_conversation_flow(json_file)
    # for item in result4:
    #     print(f"[{item['start']:.1f}s] {item['speaker']}: {item['text']}")
    
    # To use with your file:
    # speaker_texts = load_and_extract_from_file("your_file.json")
    # print(speaker_texts)

print(load_and_extract_from_file("transcription_test_results_26-06-2025 15:08:17.json"))