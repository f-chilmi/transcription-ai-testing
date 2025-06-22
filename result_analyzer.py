import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from config import OUTPUT_CONFIG

class ResultsAnalyzer:
    """Analyze and visualize transcription test results"""
    
    def __init__(self, results_file: str = OUTPUT_CONFIG.results_filename):
        self.results_file = results_file
        self.results = self.load_results()
    
    def load_results(self):
        """Load results from JSON file"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in results file: {self.results_file}")
            return None
    
    def get_performance_summary(self):
        """Get performance summary across all tests"""
        if not self.results:
            return None
        
        summary = {
            'fastest_method': {},
            'most_efficient': {},
            'best_accuracy': {},
            'resource_usage': {}
        }
        
        for audio_name, file_results in self.results.get('results', {}).items():
            times = []
            cpu_usage = []
            accuracy_scores = []
            
            for test_name, result in file_results.items():
                if result.get('success', False):
                    time_taken = result.get('processing_time', float('inf'))
                    times.append((test_name, time_taken))
                    
                    # CPU efficiency (lower average usage = more efficient)
                    resource_usage = result.get('resource_usage', {})
                    avg_cpu = resource_usage.get('avg_cpu_per_core', [])
                    if avg_cpu:
                        avg_cpu_usage = sum(avg_cpu) / len(avg_cpu)
                        cpu_usage.append((test_name, avg_cpu_usage))
                    
                    # Accuracy proxy (more speakers detected = better diarization)
                    speakers = result.get('speakers_detected', 0)
                    accuracy_scores.append((test_name, speakers))
            
            # Find fastest method
            if times:
                fastest = min(times, key=lambda x: x[1])
                summary['fastest_method'][audio_name] = {
                    'method': fastest[0],
                    'time': fastest[1],
                    'speedup_vs_slowest': max(times, key=lambda x: x[1])[1] / fastest[1]
                }
            
            # Find most CPU efficient
            if cpu_usage:
                most_efficient = min(cpu_usage, key=lambda x: x[1])
                summary['most_efficient'][audio_name] = {
                    'method': most_efficient[0],
                    'cpu_usage': most_efficient[1]
                }
            
            # Find best accuracy (most speakers detected)
            if accuracy_scores:
                best_accuracy = max(accuracy_scores, key=lambda x: x[1])
                summary['best_accuracy'][audio_name] = {
                    'method': best_accuracy[0],
                    'speakers_detected': best_accuracy[1]
                }
        
        return summary
    
    def get_thread_scaling_analysis(self):
        """Analyze thread scaling performance"""
        if not self.results:
            return None
        
        scaling_data = {}
        
        for audio_name, file_results in self.results.get('results', {}).items():
            if 'thread_scaling' in file_results:
                thread_results = file_results['thread_scaling'].get('results', {})
                
                threads = []
                times = []
                cpu_usage = []
                
                for thread_count, result in thread_results.items():
                    if result.get('success', False):
                        threads.append(int(thread_count))
                        times.append(result.get('processing_time', 0))
                        
                        resource_usage = result.get('resource_usage', {})
                        avg_cpu = resource_usage.get('avg_cpu_per_core', [])
                        if avg_cpu:
                            cpu_usage.append(sum(avg_cpu) / len(avg_cpu))
                        else:
                            cpu_usage.append(0)
                
                if threads:
                    scaling_data[audio_name] = {
                        'threads': threads,
                        'times': times,
                        'cpu_usage': cpu_usage,
                        'efficiency': [t / times[0] if times[0] > 0 else 0 for t in times]  # Relative to single thread
                    }
        
        return scaling_data
    
    def print_recommendations(self):
        """Print actionable recommendations based on results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print("\n🎯 PERFORMANCE RECOMMENDATIONS")
        print("=" * 60)
        
        summary = self.get_performance_summary()
        scaling_data = self.get_thread_scaling_analysis()
        
        # Speed recommendations
        print("\n🚀 SPEED OPTIMIZATION:")
        for audio_name, fastest_info in summary['fastest_method'].items():
            method = fastest_info['method']
            time = fastest_info['time']
            speedup = fastest_info['speedup_vs_slowest']
            
            print(f"  📁 {audio_name}: Use '{method}' ({time:.1f}s, {speedup:.1f}x faster)")
            
            if 'whisper_only' in method:
                print(f"     💡 Consider this for real-time applications")
            elif 'baseline' in method:
                print(f"     💡 Best for accuracy-critical applications")
        
        # Resource optimization
        print(f"\n⚡ RESOURCE OPTIMIZATION:")
        for audio_name, efficient_info in summary['most_efficient'].items():
            method = efficient_info['method']
            cpu_usage = efficient_info['cpu_usage']
            
            print(f"  📁 {audio_name}: '{method}' uses {cpu_usage:.1f}% CPU")
            
            if cpu_usage < 50:
                print(f"     💡 Consider increasing threads or batch size")
            elif cpu_usage > 90:
                print(f"     💡 Consider reducing threads to prevent overload")
        
        # Thread scaling recommendations
        if scaling_data:
            print(f"\n🔧 THREAD OPTIMIZATION:")
            for audio_name, data in scaling_data.items():
                threads = data['threads']
                times = data['times']
                
                # Find optimal thread count (best time improvement per thread)
                if len(threads) > 1:
                    improvements = []
                    for i in range(1, len(threads)):
                        improvement = (times[i-1] - times[i]) / times[i-1]
                        improvements.append((threads[i], improvement))
                    
                    best_thread_count = max(improvements, key=lambda x: x[1])[0]
                    print(f"  📁 {audio_name}: Optimal thread count = {best_thread_count}")
        
        # Accuracy recommendations
        print(f"\n🎯 ACCURACY OPTIMIZATION:")
        for audio_name, accuracy_info in summary['best_accuracy'].items():
            method = accuracy_info['method']
            speakers = accuracy_info['speakers_detected']
            
            print(f"  📁 {audio_name}: '{method}' detected {speakers} speakers")
            
            if speakers == 0:
                print(f"     ⚠️  No speakers detected - check audio quality")
            elif 'baseline' in method:
                print(f"     ✅ Good diarization performance")
        
        # Overall recommendations
        print(f"\n📋 FINAL RECOMMENDATIONS:")
        
        # Find the best overall method
        all_methods = {}
        for audio_results in summary['fastest_method'].values():
            method = audio_results['method']
            all_methods[method] = all_methods.get(method, 0) + 1
        
        if all_methods:
            most_common = max(all_methods.items(), key=lambda x: x[1])
            print(f"  🏆 Best overall method: {most_common[0]} (fastest on {most_common[1]} files)")
        
        # System utilization
        system_info = self.results.get('system_info', {})
        cpu_count = system_info.get('cpu_count', 0)
        memory_gb = system_info.get('memory_total_gb', 0)
        
        print(f"  💻 System: {cpu_count} cores, {memory_gb:.1f}GB RAM")
        
        if cpu_count >= 6:
            print(f"     💡 Your system can handle full diarization (baseline method)")
        else:
            print(f"     💡 Consider whisper-only method for better performance")
    
    def export_csv_report(self, filename: str = "performance_report.csv"):
        """Export performance data to CSV for further analysis"""
        if not self.results:
            print("❌ No results to export")
            return
        
        rows = []
        
        for audio_name, file_results in self.results.get('results', {}).items():
            for test_name, result in file_results.items():
                if result.get('success', False):
                    resource_usage = result.get('resource_usage', {})
                    avg_cpu = resource_usage.get('avg_cpu_per_core', [])
                    avg_cpu_usage = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                    
                    row = {
                        'audio_file': audio_name,
                        'test_method': test_name,
                        'processing_time': result.get('processing_time', 0),
                        'segments_count': result.get('segments_count', 0),
                        'speakers_detected': result.get('speakers_detected', 0),
                        'avg_cpu_usage': avg_cpu_usage,
                        'max_memory_usage': resource_usage.get('max_memory', 0),
                        'threads': result.get('threads', 0),
                        'success': True
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"📊 Performance report exported to: {filename}")
        else:
            print("❌ No successful results to export")
    
    def create_performance_chart(self, save_path: str = "performance_chart.png"):
        """Create a performance comparison chart"""
        if not self.results:
            print("❌ No results to chart")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Collect data for chart
            methods = []
            times = []
            audio_files = []
            
            for audio_name, file_results in self.results.get('results', {}).items():
                for test_name, result in file_results.items():
                    if result.get('success', False):
                        methods.append(test_name)
                        times.append(result.get('processing_time', 0))
                        audio_files.append(audio_name)
            
            if not methods:
                print("❌ No data to chart")
                return
            
            # Create chart
            plt.figure(figsize=(12, 8))
            
            # Group by method
            method_data = {}
            for method, time, audio in zip(methods, times, audio_files):
                if method not in method_data:
                    method_data[method] = []
                method_data[method].append(time)
            
            # Plot bars
            method_names = list(method_data.keys())
            avg_times = [sum(times)/len(times) for times in method_data.values()]
            
            bars = plt.bar(range(len(method_names)), avg_times)
            plt.xlabel('Method')
            plt.ylabel('Average Processing Time (seconds)')
            plt.title('Transcription Method Performance Comparison')
            plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, time in zip(bars, avg_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time:.1f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance chart saved to: {save_path}")
            
        except ImportError:
            print("📊 Install matplotlib to generate charts: pip install matplotlib")
        except Exception as e:
            print(f"❌ Error creating chart: {str(e)}")

# Quick analysis function
def analyze_results(results_file: str = OUTPUT_CONFIG.results_filename):
    """Quick function to analyze results"""
    analyzer = ResultsAnalyzer(results_file)
    
    if analyzer.results:
        analyzer.print_recommendations()
        analyzer.export_csv_report()
        analyzer.create_performance_chart()
        
        print(f"\n✅ Analysis complete! Check:")
        print(f"  📊 performance_report.csv")
        print(f"  📈 performance_chart.png")
    else:
        print("❌ No results file found. Run tests first!")

if __name__ == "__main__":
    analyze_results()