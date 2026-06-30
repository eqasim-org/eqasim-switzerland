import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_file_extension(filepath):
    """Get file extension/format."""
    ext = Path(filepath).suffix.lower()
    return ext if ext else "No extension"


def get_zip_uncompressed_size(filepath):
    """
    Get the uncompressed size of a zip file without extracting it.
    Returns None if not a valid zip file or if reading fails.
    """
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            # Sum up all uncompressed sizes in the archive
            total_uncompressed = sum(info.file_size for info in zip_ref.infolist())
            return total_uncompressed
    except (zipfile.BadZipFile, Exception):
        return None


def is_hidden_or_cached(dirname):
    """Check if a directory is hidden or cached (starts with .)."""
    return dirname.startswith('.')


def is_backup_file(filename):
    """Check if a file is a backup file (contains '.backup.' in name)."""
    return '.backup.' in filename.lower()


def analyze_folder(root_path):
    """Analyze all files in the folder structure."""
    file_tree = []
    total_files = 0
    total_size = 0
    total_uncompressed_size = 0
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out hidden/cached directories (modify in-place to prevent recursion)
        dirnames[:] = [d for d in dirnames if not is_hidden_or_cached(d)]
        
        # Calculate relative path from root
        rel_path = os.path.relpath(dirpath, root_path)
        
        # Skip if this is a hidden directory
        if rel_path != '.' and is_hidden_or_cached(os.path.basename(dirpath)):
            continue
        
        # Add directory to tree
        if rel_path == '.':
            indent_level = 0
            display_name = os.path.basename(os.path.abspath(root_path))
        else:
            indent_level = rel_path.count(os.sep) + 1
            display_name = os.path.basename(dirpath)
        
        # Sort directories and files for consistent output
        dirnames.sort()
        filenames.sort()
        
        # Filter out hidden files and backup files
        visible_filenames = [
            f for f in filenames 
            if not f.startswith('.') and not is_backup_file(f)
        ]
        
        # Add directory entry
        file_tree.append({
            'type': 'directory',
            'name': display_name,
            'path': dirpath,
            'rel_path': rel_path,
            'indent_level': indent_level,
            'size': 0,
            'extension': '',
            'uncompressed_size': None,
            'file_count': len(visible_filenames)
        })
        
        # Process files in this directory
        for filename in visible_filenames:
            filepath = os.path.join(dirpath, filename)
            
            try:
                # Get file size
                file_size = os.path.getsize(filepath)
                file_ext = get_file_extension(filepath)
                
                # Check if it's a zip file and get uncompressed size
                uncompressed_size = None
                if file_ext == '.zip':
                    uncompressed_size = get_zip_uncompressed_size(filepath)
                
                file_info = {
                    'type': 'file',
                    'name': filename,
                    'path': filepath,
                    'rel_path': os.path.relpath(filepath, root_path),
                    'indent_level': indent_level + 1,
                    'size': file_size,
                    'extension': file_ext,
                    'uncompressed_size': uncompressed_size
                }
                
                file_tree.append(file_info)
                total_files += 1
                total_size += file_size
                
                # Add to total uncompressed size
                if uncompressed_size is not None:
                    total_uncompressed_size += uncompressed_size
                else:
                    total_uncompressed_size += file_size
                
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not access {filepath}: {e}")
    
    return file_tree, total_files, total_size, total_uncompressed_size


def generate_text_report(file_tree, total_files, total_size, total_uncompressed_size, root_path, output_path):
    """Generate a text report of the file tree."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header with summary
        f.write("=" * 80 + "\n")
        f.write("FOLDER STRUCTURE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root Path: {os.path.abspath(root_path)}\n")
        f.write("-" * 80 + "\n")
        
        # Summary section
        f.write("SUMMARY:\n")
        f.write(f"  Total Files: {total_files}\n")
        f.write(f"  Total Folder Size (compressed): {format_size(total_size)} ({total_size:,} bytes)\n")
        f.write(f"  Total Folder Size (uncompressed): {format_size(total_uncompressed_size)} ({total_uncompressed_size:,} bytes)\n")
        
        if total_uncompressed_size > total_size:
            savings = total_uncompressed_size - total_size
            ratio = (savings / total_uncompressed_size) * 100
            f.write(f"  Space Saved by Compression: {format_size(savings)} ({ratio:.1f}%)\n")
        
        f.write("-" * 80 + "\n")
        
        # Summary by file type
        f.write("\nSUMMARY BY FILE TYPE:\n")
        f.write("-" * 80 + "\n")
        
        # Count files by extension
        ext_stats = {}
        for item in file_tree:
            if item['type'] == 'file':
                ext = item['extension']
                if ext not in ext_stats:
                    ext_stats[ext] = {'count': 0, 'size': 0, 'uncompressed_size': 0}
                ext_stats[ext]['count'] += 1
                ext_stats[ext]['size'] += item['size']
                if item['uncompressed_size'] is not None:
                    ext_stats[ext]['uncompressed_size'] += item['uncompressed_size']
                else:
                    ext_stats[ext]['uncompressed_size'] += item['size']
        
        # Sort by count (descending)
        sorted_exts = sorted(ext_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for ext, stats in sorted_exts:
            uncomp_str = f" | Uncompressed: {format_size(stats['uncompressed_size'])}" if stats['uncompressed_size'] != stats['size'] else ""
            f.write(f"{ext:15s}: {stats['count']:5d} files, {format_size(stats['size']):>10s}{uncomp_str}\n")
        
        f.write("=" * 80 + "\n\n")
        
        # File tree
        f.write("FILE TREE:\n")
        f.write("-" * 80 + "\n")
        
        # Group files by directory for limiting display
        current_dir = None
        dir_files = []
        items_to_display = []
        
        for item in file_tree:
            if item['type'] == 'directory':
                # If we were collecting files for previous directory, process them
                if current_dir is not None and dir_files:
                    items_to_display.extend(process_directory_files(current_dir, dir_files))
                
                # Start new directory
                current_dir = item
                dir_files = []
                items_to_display.append(item)
            else:
                dir_files.append(item)
        
        # Process last directory
        if current_dir is not None and dir_files:
            items_to_display.extend(process_directory_files(current_dir, dir_files))
        
        # Display all items
        for item in items_to_display:
            if item['type'] == 'directory':
                indent = "  " * item['indent_level']
                f.write(f"{indent}📁 {item['name']}/ ({item['file_count']} files)\n")
            elif item.get('is_summary'):
                # This is a summary line for truncated directory
                indent = "  " * item['indent_level']
                f.write(f"{indent}   ... [{item['remaining_count']} more files, {format_size(item['remaining_size'])}]\n")
            else:
                indent = "  " * item['indent_level']
                size_str = format_size(item['size'])
                ext_info = f"[{item['extension']}]"
                
                # Add uncompressed size info for zip files
                zip_info = ""
                if item['uncompressed_size'] is not None:
                    uncomp_str = format_size(item['uncompressed_size'])
                    compression_ratio = (1 - item['size'] / item['uncompressed_size']) * 100 if item['uncompressed_size'] > 0 else 0
                    zip_info = f" | Uncompressed: {uncomp_str} ({compression_ratio:.1f}% compressed)"
                
                f.write(f"{indent}📄 {item['name']} {ext_info} ({size_str}){zip_info}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Text report saved to: {output_path}")


def process_directory_files(dir_item, files):
    """Process files in a directory, limiting to first 10 if there are more."""
    result = []
    
    if len(files) <= 10:
        # Show all files
        result.extend(files)
    else:
        # Show first 10 files
        result.extend(files[:10])
        
        # Calculate remaining
        remaining_files = files[10:]
        remaining_count = len(remaining_files)
        remaining_size = sum(f['size'] for f in remaining_files)
        
        # Add summary line
        summary_item = {
            'type': 'file',
            'is_summary': True,
            'indent_level': files[0]['indent_level'],
            'remaining_count': remaining_count,
            'remaining_size': remaining_size
        }
        result.append(summary_item)
    
    return result


def generate_csv_report(file_tree, total_files, total_size, total_uncompressed_size, root_path, output_path):
    """Generate a CSV report of the file tree."""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Type', 
            'Name', 
            'Relative Path', 
            'Size (bytes)', 
            'Size (human)', 
            'Extension',
            'Uncompressed Size (bytes)',
            'Uncompressed Size (human)'
        ])
        
        # Data rows
        for item in file_tree:
            if item['type'] == 'file':
                size_human = format_size(item['size'])
                uncomp_size = item['uncompressed_size'] if item['uncompressed_size'] is not None else ''
                uncomp_human = format_size(uncomp_size) if uncomp_size else ''
                
                writer.writerow([
                    item['type'],
                    item['name'],
                    item['rel_path'],
                    item['size'],
                    size_human,
                    item['extension'],
                    uncomp_size,
                    uncomp_human
                ])
            elif item['type'] == 'directory':
                writer.writerow([
                    item['type'],
                    item['name'],
                    item['rel_path'],
                    '',
                    '',
                    '',
                    '',
                    ''
                ])
    
    print(f"CSV report saved to: {output_path}")


def main():
    """Main function to run the folder analysis."""
    if len(sys.argv) < 2:
        print("Usage: python folder_analyzer.py <folder_path> [output_format]")
        print("Output formats: txt (default), csv, both")
        sys.exit(1)
    
    root_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'txt'
    
    # Validate path
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(root_path):
        print(f"Error: Path '{root_path}' is not a directory.")
        sys.exit(1)
    
    print(f"Analyzing folder: {os.path.abspath(root_path)}")
    print("This may take a moment for large directories...")
    print("Note: Reading zip file headers to get uncompressed sizes...")
    print("Note: Excluding hidden folders, backup files, and cached directories...\n")
    
    # Analyze the folder
    file_tree, total_files, total_size, total_uncompressed_size = analyze_folder(root_path)
    
    print(f"\nAnalysis complete!")
    print(f"Total files found: {total_files}")
    print(f"Total folder size (compressed): {format_size(total_size)} ({total_size:,} bytes)")
    print(f"Total folder size (uncompressed): {format_size(total_uncompressed_size)} ({total_uncompressed_size:,} bytes)")
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.basename(os.path.abspath(root_path))
    
    if output_format in ['txt', 'both']:
        txt_output = f"folder_analysis_{base_name}_{timestamp}.txt"
        generate_text_report(file_tree, total_files, total_size, total_uncompressed_size, root_path, txt_output)
    
    if output_format in ['csv', 'both']:
        csv_output = f"folder_analysis_{base_name}_{timestamp}.csv"
        generate_csv_report(file_tree, total_files, total_size, total_uncompressed_size, root_path, csv_output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()