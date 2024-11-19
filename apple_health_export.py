import xml.etree.ElementTree as ET
from datetime import datetime
import csv
from collections import defaultdict
import zipfile
import os
from pathlib import Path

class AppleHealthProcessor:
    def __init__(self, xml_path, output_path=None):
        """
        Initialize processor with input XML path and optional output path
        """
        self.xml_path = Path(xml_path)
        self.output_path = Path(output_path) if output_path else Path.cwd()

    def extract_date_range(self, export_root):
        """Extract the min and max dates from the health data"""
        min_date = None
        max_date = None

        for record in export_root.findall('Record'):
            start_date_str = record.get('startDate')
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S %z')

            if min_date is None or start_date < min_date:
                min_date = start_date
            if max_date is None or start_date > max_date:
                max_date = start_date

        return min_date, max_date

    def export_data_by_month(self, export_root, year):
        """Export data for a specific year into monthly CSV files"""
        records_by_month = defaultdict(list)

        for record in export_root.findall('Record'):
            start_date_str = record.get('startDate')
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S %z')

            if start_date.year == year:
                month_key = start_date.strftime('%Y-%m')
                records_by_month[month_key].append(record)

        filenames = []
        for month_key, records in records_by_month.items():
            filename = self.output_path / f"{month_key}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['RecordType', 'StartDate', 'Value'])
                for record in records:
                    record_type = record.get('type')
                    start_date_str = record.get('startDate')
                    value = record.get('value')
                    writer.writerow([record_type, start_date_str, value])
            filenames.append(filename)
        return filenames

    def zip_files(self, filenames, zip_filename):
        """Zip the CSV files and cleanup"""
        zip_path = self.output_path / zip_filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in filenames:
                zipf.write(file, file.name)
                os.remove(file)  # Remove the CSV file after adding to zip
        return zip_path

    def process(self):
        """Main processing method"""
        # Parse XML
        export = ET.parse(self.xml_path)
        export_root = export.getroot()

        # Get date range
        min_date, max_date = self.extract_date_range(export_root)
        print(f"Processing data from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        # Process each year
        all_filenames = []
        for year in range(min_date.year, max_date.year + 1):
            print(f"Processing year {year}...")
            filenames = self.export_data_by_month(export_root, year)
            all_filenames.extend(filenames)

        # Create zip file
        if all_filenames:
            zip_filename = f"apple_health_data_{min_date.year}-{max_date.year}.zip"
            zip_path = self.zip_files(all_filenames, zip_filename)
            print(f"Data exported to {zip_path}")
            return zip_path
        else:
            print("No data to export")
            return None

def main(xml_path, output_path=None):
    """Convenience function to process Apple Health data"""
    processor = AppleHealthProcessor(xml_path, output_path)
    return processor.process()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        main(xml_path, output_path)
    else:
        print("Please provide the path to the Apple Health Export XML file")
        print("Usage: python script.py <xml_path> [output_path]")
