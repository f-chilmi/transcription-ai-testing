import json
from typing import List, Dict, Any, Union


class TranscriptFormatter:
    """
    A utility class to format transcript JSON data into readable conversation format.
    Supports both simple segment arrays and complex nested JSON structures.
    """
    
    def __init__(self):
        self.grouped_transcript = []
    
    def extract_segments(self, data: Union[Dict, List]) -> List[Dict]:
        """
        Extract segments from various JSON structures.
        
        Args:
            data: JSON data that can be a list of segments or nested structure
            
        Returns:
            List of segment dictionaries with 'text' and 'speaker' keys
        """
        segments = []
        
        if isinstance(data, list):
            # Direct array of segments
            segments = data
        elif isinstance(data, dict):
            # Check various possible nested structures
            if 'result' in data:
                segments = data['result']
            elif 'results' in data:
                results = data['results']
                if isinstance(results, dict):
                    # Look for segments in nested results
                    if 'batch_processing' in results and 'results' in results['batch_processing']:
                        batch_results = results['batch_processing']['results']
                        if isinstance(batch_results, list) and len(batch_results) > 0:
                            # Get first result's segments
                            first_result = batch_results[0]
                            # first_result = batch_results[0]['diarization_whisper']
                            for key, value in first_result.items():
                                print(43, key, key is 'segments', isinstance(value, dict) and 'segments' in value)
                                if isinstance(value, dict) and 'segments' in value:
                                    segments = value['segments']
                                    break
                    elif 'segments' in results:
                        segments = results['segments']
                elif isinstance(results, list):
                    segments = results
            elif 'segments' in data:
                segments = data['segments']
        
        return segments
    
    def group_by_speaker(self, segments: List[Dict]) -> List[Dict]:
        """
        Group consecutive segments by the same speaker.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of grouped segments with combined text
        """
        if not segments:
            return []
        
        grouped = []
        current_group = None
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            
            if not text:
                continue
                
            if current_group is None or current_group['speaker'] != speaker:
                # Start new group
                current_group = {
                    'speaker': speaker,
                    'text': text,
                    'start': segment.get('start'),
                    'end': segment.get('end')
                }
                grouped.append(current_group)
            else:
                # Add to existing group
                current_group['text'] += ' ' + text
                current_group['end'] = segment.get('end')  # Update end time
        
        return grouped
    
    def format_conversation(self, grouped_segments: List[Dict], 
                          speaker_names: Dict[str, str] = None) -> str:
        """
        Format grouped segments into readable conversation.
        
        Args:
            grouped_segments: List of grouped segments
            speaker_names: Optional mapping of speaker IDs to names
            
        Returns:
            Formatted conversation string
        """
        if not grouped_segments:
            return "No conversation data found."
        
        formatted_lines = []
        
        for group in grouped_segments:
            speaker = group['speaker']
            text = group['text']
            
            # Use custom speaker name if provided
            if speaker_names and speaker in speaker_names:
                speaker_display = speaker_names[speaker]
            else:
                speaker_display = speaker
            
            formatted_lines.append(f"{speaker_display}: {text}")
        
        return '\n\n'.join(formatted_lines)
    
    def format_from_json_file(self, file_path: str, 
                             speaker_names: Dict[str, str] = None) -> str:
        """
        Load JSON file and format it into readable conversation.
        
        Args:
            file_path: Path to JSON file
            speaker_names: Optional mapping of speaker IDs to names
            
        Returns:
            Formatted conversation string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.format_from_data(data, speaker_names)
            
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def format_from_data(self, data: Union[Dict, List], 
                        speaker_names: Dict[str, str] = None) -> str:
        """
        Format JSON data into readable conversation.
        
        Args:
            data: JSON data (dict or list)
            speaker_names: Optional mapping of speaker IDs to names
            
        Returns:
            Formatted conversation string
        """
        try:
            segments = self.extract_segments(data)
            
            if not segments:
                return "No segments found in the data."
            
            grouped = self.group_by_speaker(segments)
            
            if not grouped:
                return "No valid conversation data found."
            
            return self.format_conversation(grouped, speaker_names)
            
        except Exception as e:
            return f"Error formatting data: {str(e)}"
    
    def save_formatted_conversation(self, formatted_text: str, output_path: str):
        """
        Save formatted conversation to a file.
        
        Args:
            formatted_text: The formatted conversation string
            output_path: Path where to save the formatted text
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"Formatted conversation saved to: {output_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
    
    def get_conversation_stats(self, grouped_segments: List[Dict]) -> Dict[str, Any]:
        """
        Get statistics about the conversation.
        
        Args:
            grouped_segments: List of grouped segments
            
        Returns:
            Dictionary with conversation statistics
        """
        if not grouped_segments:
            return {}
        
        speakers = set(group['speaker'] for group in grouped_segments)
        total_words = sum(len(group['text'].split()) for group in grouped_segments)
        
        # Calculate duration if start/end times are available
        duration = None
        if all('start' in group and 'end' in group for group in grouped_segments):
            start_time = min(group['start'] for group in grouped_segments if group['start'] is not None)
            end_time = max(group['end'] for group in grouped_segments if group['end'] is not None)
            if start_time is not None and end_time is not None:
                duration = end_time - start_time
        
        return {
            'total_speaker_turns': len(grouped_segments),
            'unique_speakers': len(speakers),
            'speakers': list(speakers),
            'total_words': total_words,
            'duration_seconds': duration
        }


# Convenience functions for quick usage
def format_transcript_file(file_path: str, output_path: str = None, 
                          speaker_names: Dict[str, str] = None) -> str:
    """
    Quick function to format a transcript file.
    
    Args:
        file_path: Path to JSON transcript file
        output_path: Optional path to save formatted output
        speaker_names: Optional mapping of speaker IDs to names
        
    Returns:
        Formatted conversation string
    """
    formatter = TranscriptFormatter()
    formatted = formatter.format_from_json_file(file_path, speaker_names)
    
    if output_path:
        formatter.save_formatted_conversation(formatted, output_path)
    
    return formatted


def format_transcript_data(data: Union[Dict, List], 
                          speaker_names: Dict[str, str] = None) -> str:
    """
    Quick function to format transcript data.
    
    Args:
        data: JSON data (dict or list)
        speaker_names: Optional mapping of speaker IDs to names
        
    Returns:
        Formatted conversation string
    """
    formatter = TranscriptFormatter()
    return formatter.format_from_data(data, speaker_names)


# Example usage
if __name__ == "__main__":
    # Example 1: Format from file
    # formatted = format_transcript_file(
    #     'transcript.json',
    #     'formatted_conversation.txt',
    #     speaker_names={'SPEAKER_00': 'Phil', 'SPEAKER_01': 'Georgie'}
    # )
    # print(formatted)
    
    # Example 2: Format from data
    formatter = TranscriptFormatter()
    
    # Sample data structure (like your JSON)
    sample_data = {
        "results": {
            "batch_processing": {
                "results": [
                    {
                        "diarization_whisper": {
                            "segments": [
                                {
                                    "start": 0.02,
                                    "end": 0.543,
                                    "text": "Hello",
                                    "speaker": "SPEAKER_00"
                                },
                                {
                                    "start": 0.583,
                                    "end": 1.127,
                                    "text": "world",
                                    "speaker": "SPEAKER_00"
                                },
                                {
                                    "start": 1.2,
                                    "end": 1.8,
                                    "text": "How are you?",
                                    "speaker": "SPEAKER_01"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }
    
    formatted = formatter.format_from_data(
        sample_data, 
        speaker_names={'SPEAKER_00': 'Alice', 'SPEAKER_01': 'Bob'}
    )
    print(formatted)
    
    # Get conversation statistics
    segments = formatter.extract_segments(sample_data)
    grouped = formatter.group_by_speaker(segments)
    stats = formatter.get_conversation_stats(grouped)
    print(f"\nConversation Stats: {stats}")