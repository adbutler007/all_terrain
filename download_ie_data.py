#!/usr/bin/env python3
"""
Download the updated Shiller ie_data.xls file.

This downloads from shillerdata.com which maintains updated versions,
rather than the stale Yale/Shiller site which only has data through 2023.
"""

import os
import sys
import re
import requests
from datetime import datetime


def get_latest_download_url(verbose=True):
    """
    Fetch the latest download URL from shillerdata.com.

    Returns
    -------
    str or None
        The download URL if found, None otherwise
    """
    try:
        if verbose:
            print("Fetching latest download URL from shillerdata.com...")

        response = requests.get("https://shillerdata.com", timeout=10)
        response.raise_for_status()

        # Look for the download link pattern
        # The link format is: //img1.wsimg.com/blobby/go/.../ie_data.xls?ver=...
        pattern = r'//img1\.wsimg\.com/blobby/go/[^"\']+?ie_data\.xls[^"\'>]*'
        match = re.search(pattern, response.text)

        if match:
            url = match.group(0)
            # Add https: if it starts with //
            if url.startswith('//'):
                url = 'https:' + url
            if verbose:
                print(f"Found download URL: {url}")
            return url
        else:
            if verbose:
                print("Could not find download link on shillerdata.com")
            return None

    except Exception as e:
        if verbose:
            print(f"Error fetching download URL: {e}")
        return None


def download_ie_data(output_file="ie_data.xls", verbose=True):
    """
    Download the updated ie_data.xls file.

    Parameters
    ----------
    output_file : str
        Name of the output file (default: ie_data.xls)
    verbose : bool
        Whether to print status messages

    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    # Get the latest URL from shillerdata.com
    url = get_latest_download_url(verbose)

    if not url:
        print("Failed to get download URL from shillerdata.com")
        return False

    try:
        if verbose:
            print(f"Downloading updated Shiller data...")
            print("Note: shillerdata.com maintains updated versions (Yale site only has through 2023)")

        # Download with headers to avoid potential blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Save the file
        with open(output_file, 'wb') as f:
            f.write(response.content)

        # Verify file size
        file_size = os.path.getsize(output_file)
        if verbose:
            print(f"\nSuccessfully downloaded {output_file}")
            print(f"File size: {file_size:,} bytes")
            print(f"Download timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Basic validation - file should be at least 1MB
        if file_size < 1000000:
            print("Warning: File seems unusually small. It may be corrupted or incomplete.")
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    """Main function to download the data file."""
    # Check if output file already exists
    output_file = "ie_data.xls"

    if os.path.exists(output_file):
        response = input(f"{output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)

    # Download the file
    success = download_ie_data(output_file)

    if success:
        print("\nFile downloaded successfully!")
        print("You can now run regime_plot.py to generate the plots.")
    else:
        print("\nDownload failed. Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()