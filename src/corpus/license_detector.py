"""
License detection module for Creative Commons licenses in PubMed articles.
"""
import re
import xml.etree.ElementTree as ET
from typing import Optional


class LicenseDetector:
    """Detects and categorizes Creative Commons licenses from PubMed XML."""
    
    # License information mapping
    LICENSE_INFO = {
        "cc-by-nc-nd": {
            "commercial_use": False,
            "modifications_allowed": False,
            "must_credit": True,
            "notes": "Can only share as-is, non-commercially"
        },
        "cc-by-nc-sa": {
            "commercial_use": False,
            "modifications_allowed": "same_license",
            "must_credit": True,
            "notes": "Derivatives must use same license"
        },
        "cc-by-nc": {
            "commercial_use": False,
            "modifications_allowed": True,
            "must_credit": True,
            "notes": "No obligation to license under the same terms"
        },
        "cc-by-sa": {
            "commercial_use": True,
            "modifications_allowed": "same_license",
            "must_credit": True,
            "notes": "Derivatives must use same license"
        },
        "cc-by": {
            "commercial_use": True,
            "modifications_allowed": True,
            "must_credit": True,
            "notes": "Most permissive with attribution"
        },
        "cc0": {
            "commercial_use": True,
            "modifications_allowed": True,
            "must_credit": False,
            "notes": "Public domain - no restrictions"
        }
    }
    
    @staticmethod
    def detect_cc_license(lic_elem: Optional[ET.Element]) -> str:
        """
        Inspect <license> … </license> for Creative Commons URLs or keywords
        and return a normalized string such as 'cc-by', 'cc-by-nc', 'cc0', or 'other'.
        
        Args:
            lic_elem: XML license element from PubMed article
            
        Returns:
            License type as string
        """
        if lic_elem is None:
            return "other"
        
        # Gather candidate strings: any ext-link href + full text
        candidates = []
        
        # Extract external links
        for link in lic_elem.findall(".//ext-link[@ext-link-type='uri']"):
            href = link.get("{http://www.w3.org/1999/xlink}href") or link.get("href")
            if href:
                candidates.append(href.lower())
        
        # Add full text content
        candidates.append("".join(lic_elem.itertext()).lower())
        
        # Search for CC patterns
        for text in candidates:
            if "creativecommons.org" not in text and "publicdomain" not in text:
                continue
            
            # Order matters (most restrictive first)
            if re.search(r"by[-_]nc[-_]nd", text):
                return "cc-by-nc-nd"
            if re.search(r"by[-_]nc[-_]sa", text):
                return "cc-by-nc-sa"
            if re.search(r"by[-_]nc", text):
                return "cc-by-nc"
            if re.search(r"by[-_]sa", text):
                return "cc-by-sa"
            if "/by/" in text:
                return "cc-by"
            if "publicdomain/zero" in text or "cc0" in text or "public domain" in text:
                return "cc0"
        
        return "other"
    
    @classmethod
    def get_license_info(cls, license_type: str) -> dict:
        """
        Get detailed information about a license type.
        
        Args:
            license_type: License type string
            
        Returns:
            Dictionary with license details
        """
        return cls.LICENSE_INFO.get(license_type, {
            "commercial_use": None,
            "modifications_allowed": None,
            "must_credit": None,
            "notes": "Unknown license type"
        })
    
    @classmethod
    def is_allowed_license(cls, license_type: str, allowed_licenses: set) -> bool:
        """
        Check if a license type is in the allowed set.
        
        Args:
            license_type: License type to check
            allowed_licenses: Set of allowed license types
            
        Returns:
            True if license is allowed
        """
        return license_type in allowed_licenses