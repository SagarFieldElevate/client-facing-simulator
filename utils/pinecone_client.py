"""
Pinecone client for fetching portfolio asset data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from datetime import datetime
import os
from pinecone import Pinecone

class PineconeClient:
    def __init__(self, api_key: str):
        """Initialize Pinecone client"""
        # Initialize Pinecone with the v3 API
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index("intelligence-main")
        
        # Asset name mappings in Pinecone
        self.asset_mappings = {
            'stocks': 'SPY Daily Close Price',
            'bonds': 'AGG Daily Close Price',
            'real_estate': 'VNQ Daily Close Price',
            'crypto': 'COIN50 Perpetual Index (365 Days)',
            'btc': 'BTC Daily Close Price'
        }
        
    def fetch_asset_data(self, asset_type: str, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch historical data for a specific asset from Pinecone
        
        Args:
            asset_type: One of 'stocks', 'bonds', 'real_estate', 'crypto', 'btc'
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with date index and price column
        """
        if asset_type not in self.asset_mappings:
            raise ValueError(f"Unknown asset type: {asset_type}")
        
        excel_name = self.asset_mappings[asset_type]
        
        # Query with dummy vector and filter by excel_name
        dummy_vector = [0.0] * 1536
        
        try:
            response = self.index.query(
                vector=dummy_vector,
                top_k=limit,
                filter={"excel_name": excel_name},
                include_metadata=True
            )
            
            if not response.matches:
                raise ValueError(f"No data found for {excel_name}")
            
            # Parse the data based on asset type
            data = []
            for match in response.matches:
                metadata = match.metadata
                if 'raw_text' in metadata:
                    parsed = self._parse_raw_text(metadata['raw_text'], asset_type)
                    if parsed:
                        data.append(parsed)
            
            if not data:
                raise ValueError(f"No valid data parsed for {excel_name}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Remove duplicates and forward fill missing values
            df = df[~df.index.duplicated(keep='first')]
            df = df.ffill()
            
            return df
            
        except Exception as e:
            print(f"Error fetching {asset_type} data: {str(e)}")
            raise
    
    def _parse_raw_text(self, raw_text: str, asset_type: str) -> Optional[Dict]:
        """Parse raw text based on asset type"""
        try:
            if asset_type == 'stocks':
                # Format: "Date: YYYY-MM-DD HH:MM:SS | SPY Close Price (USD): XXX.XX"
                date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', raw_text)
                close_match = re.search(r'SPY Close Price \(USD\):\s*([\d.]+)', raw_text)
                
                if date_match and close_match:
                    return {
                        'date': date_match.group(1),
                        'close': float(close_match.group(1))
                    }
                    
            elif asset_type == 'bonds':
                # Format: "Date: 2017-10-26 00:00:00 | AGG Close Price (USD): 88.17"
                date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', raw_text)
                price_match = re.search(r'AGG Close Price \(USD\):\s*([\d.]+)', raw_text)
                
                if date_match and price_match:
                    return {
                        'date': date_match.group(1),
                        'close': float(price_match.group(1))
                    }
                    
            elif asset_type == 'real_estate':
                # Format: "Date: 2019-11-26 00:00:00 | VNQ Close Price (USD): 74.75"
                date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', raw_text)
                price_match = re.search(r'VNQ Close Price \(USD\):\s*([\d.]+)', raw_text)
                
                if date_match and price_match:
                    return {
                        'date': date_match.group(1),
                        'close': float(price_match.group(1))
                    }
                    
            elif asset_type == 'crypto':
                # Format: "Date: 2024-06-22 00:00:00 | COIN50 Index Value: 254.06"
                date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', raw_text)
                value_match = re.search(r'COIN50 Index Value:\s*([\d.]+)', raw_text)
                
                if date_match and value_match:
                    return {
                        'date': date_match.group(1),
                        'close': float(value_match.group(1))
                    }
            elif asset_type == 'btc':
                # Try multiple BTC formats
                date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', raw_text)
                value_match = (
                    re.search(r'BTC Close Price \(USD\):\s*([\d.]+)', raw_text)
                    or re.search(r'Bitcoin Close Price \(USD\):\s*([\d.]+)', raw_text)
                    or re.search(r'BTC Price \(USD\):\s*([\d.]+)', raw_text)
                )
                if date_match and value_match:
                    return {
                        'date': date_match.group(1),
                        'close': float(value_match.group(1))
                    }
                    
        except Exception as e:
            print(f"Error parsing raw text: {e}")
            
        return None
    
    def fetch_all_assets(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all asset types, combining BTC with COIN50 for extended crypto history"""
        all_data = {}
        
        # First fetch traditional assets
        for asset_type in ['stocks', 'bonds', 'real_estate']:
            try:
                print(f"Fetching {asset_type} data...")
                df = self.fetch_asset_data(asset_type)
                all_data[asset_type] = df
                print(f"✓ Loaded {len(df)} records for {asset_type}")
            except Exception as e:
                print(f"✗ Error loading {asset_type}: {str(e)}")
        
        # Fetch crypto sources
        crypto_df = None
        btc_df = None
        try:
            print("Fetching crypto data (COIN50)...")
            crypto_df = self.fetch_asset_data('crypto')
            print(f"✓ Loaded {len(crypto_df)} records for COIN50")
        except Exception as e:
            print(f"✗ Error loading COIN50: {str(e)}")
        
        try:
            print("Fetching BTC data...")
            btc_df = self.fetch_asset_data('btc')
            print(f"✓ Loaded {len(btc_df)} records for BTC")
        except Exception as e:
            print(f"✗ Error loading BTC: {str(e)}")
        
        # Combine: prefer COIN50 where available; use BTC for earlier dates
        if crypto_df is not None and btc_df is not None:
            combined = btc_df.copy()
            # Overwrite with COIN50 where both exist or where COIN50 exists alone
            combined = combined.combine_first(crypto_df)  # BTC fills missing; then prefer crypto where NaN in BTC
            combined.update(crypto_df)  # ensure COIN50 overwrites overlapping dates
            all_data['crypto'] = combined.sort_index()
        elif crypto_df is not None:
            all_data['crypto'] = crypto_df.sort_index()
        elif btc_df is not None:
            all_data['crypto'] = btc_df.sort_index()
        else:
            print("✗ No crypto series available")
        
        return all_data