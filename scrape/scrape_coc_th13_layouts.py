#!/usr/bin/env python
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Create output directory if it doesn't exist
os.makedirs('th_13', exist_ok=True)

# Function to scrape layouts


def scrape_layouts(base_type, total_pages):
    # Base URL and parameters
    base_url = f'https://clashofclans-layouts.com/plans/th_13/{base_type}_{{}}.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Counter for successful downloads
    successful_downloads = 0

    for page_num in range(1, total_pages + 1):
        # Construct the page URL
        page_url = base_url.format(page_num)

        print(
            f"Processing TH13 {base_type} page {page_num}/{total_pages}: {page_url}")

        try:
            # Get the page content
            response = requests.get(page_url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'lxml')

            # Construct the expected image path pattern for this page number
            expected_img_path = f"/pics/th13_plans/{base_type}/original/th13_{base_type}_{page_num}.jpg"

            # Find the a element with the matching href
            a_tag = soup.find(
                'a', href=lambda href: href and expected_img_path in href)

            if a_tag and a_tag.find('img'):
                # Get the full URL of the image
                img_url = urljoin(
                    'https://clashofclans-layouts.com', expected_img_path)

                # File path to save the image
                file_path = f"th_13/th13_{base_type}_{page_num}.jpg"

                # Download the image
                print(f"Downloading: {img_url}")
                img_response = requests.get(img_url, headers=headers)
                img_response.raise_for_status()

                # Save the image
                with open(file_path, 'wb') as f:
                    f.write(img_response.content)

                successful_downloads += 1
                print(f"Successfully downloaded: {file_path}")
            else:
                print(f"Could not find the image on page {page_num}")

            # Add a small delay to avoid overloading the server
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page_num}: {e}")
        except Exception as e:
            print(f"An error occurred on page {page_num}: {e}")

    print(
        f"Scraping complete for {base_type}. Successfully downloaded {successful_downloads}/{total_pages} images.")
    return successful_downloads


# Scrape TH13 war layouts
print("Starting to scrape Town Hall 13 layouts...")
war_downloads = scrape_layouts('war', 130)

print(
    f"All scraping complete. Downloaded {war_downloads} TH13 images in total.")
