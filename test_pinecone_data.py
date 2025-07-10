"""
Test script to explore Pinecone data formats
This will help us understand the exact format of data for each asset type
"""
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index("intelligence-main")

# Asset mappings
asset_mappings = {
    'stocks': 'SPY Daily Close Price',
    'bonds': 'AGG Daily Close Price',
    'real_estate': 'VNQ Daily Close Price',
    'crypto': 'COIN50 Perpetual Index (365 Days)'
}

print("Testing Pinecone data fetching...\n")

# Test each asset type
for asset_type, excel_name in asset_mappings.items():
    print(f"\n{'='*60}")
    print(f"Testing {asset_type} ({excel_name})")
    print('='*60)
    
    # Query with dummy vector
    dummy_vector = [0.0] * 1536
    
    try:
        response = index.query(
            vector=dummy_vector,
            top_k=5,  # Just get 5 samples to examine format
            filter={"excel_name": excel_name},
            include_metadata=True
        )
        
        if not response.matches:
            print(f"❌ No data found for {excel_name}")
            continue
            
        print(f"✓ Found {len(response.matches)} matches")
        
        # Examine the first few matches to understand the format
        for i, match in enumerate(response.matches[:3]):
            print(f"\n--- Sample {i+1} ---")
            metadata = match.metadata
            
            # Print all metadata fields
            print("Metadata fields:", list(metadata.keys()))
            
            # Print raw_text if it exists
            if 'raw_text' in metadata:
                raw_text = metadata['raw_text']
                print(f"Raw text: {raw_text[:200]}..." if len(raw_text) > 200 else f"Raw text: {raw_text}")
                
                # Try different regex patterns
                print("\nRegex tests:")
                
                # Date patterns
                date_patterns = [
                    r'Date:\s*(\d{4}-\d{2}-\d{2})',
                    r'(\d{4}-\d{2}-\d{2})',
                    r'Date:\s*([^\|]+)',
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, raw_text)
                    if match:
                        print(f"  Date pattern '{pattern}' found: {match.group(1)}")
                        break
                
                # Price patterns
                price_patterns = [
                    r'SPY Close Price \(USD\):\s*([\d.]+)',
                    r'Close:\s*([\d.]+)',
                    r'Close Price:\s*([\d.]+)',
                    r'Price:\s*([\d.]+)',
                    r'close:\s*([\d.]+)',
                    r'(\d+\.\d+)',  # Any decimal number
                ]
                
                for pattern in price_patterns:
                    match = re.search(pattern, raw_text)
                    if match:
                        print(f"  Price pattern '{pattern}' found: {match.group(1)}")
                        break
            
            # Check for other potentially useful fields
            for key, value in metadata.items():
                if key != 'raw_text':
                    print(f"{key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error fetching {asset_type}: {str(e)}")

print("\n\nTest complete!")