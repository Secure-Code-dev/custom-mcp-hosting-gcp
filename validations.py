import re
import hashlib
import mimetypes
from typing import Dict, Any, List, Optional, Set

class ContentValidator:
    """Validates and filters file content for security and compliance"""
    
    def __init__(self):
        # Sensitive patterns to detect and redact
        self.sensitive_patterns = {
            'api_keys': [
                r'(?i)(?:api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                r'(?i)(?:secret[_-]?key|secretkey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                r'(?i)(?:access[_-]?token|accesstoken)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
            ],
            'database_urls': [
                r'(?i)(?:database[_-]?url|db[_-]?url)\s*[:=]\s*["\']?(.*?://.*?)["\']?',
                r'(?i)(?:connection[_-]?string|conn[_-]?str)\s*[:=]\s*["\']?(.*?)["\']?',
            ],
            'passwords': [
                r'(?i)password\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?',
                r'(?i)pwd\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?',
                r'(?i)pass\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?',
            ],
            'private_keys': [
                r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
                r'-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+OPENSSH\s+PRIVATE\s+KEY-----',
            ],
            'jwt_tokens': [
                r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
            ],
            'email_addresses': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'phone_numbers': [
                r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
                r'(?:\+?[1-9][0-9]{0,3}[-.\s]?)?(?:\([0-9]{1,4}\)|[0-9]{1,4})[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}',
            ],
            'credit_cards': [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            ],
            'social_security': [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b',
            ]
        }
        
        # Profanity and inappropriate content patterns
        self.inappropriate_patterns = [
            # Add your profanity filter patterns here
            r'(?i)\b(?:vulgar|obscene|inappropriate)\b',  # Placeholder - add actual patterns
        ]
        
        # File extensions that should be blocked entirely
        self.blocked_extensions = {
            '.key', '.pem', '.p12', '.pfx', '.jks', '.keystore',
            '.env', '.secret', '.credentials', '.config',
            '.exe', '.dll', '.so', '.dylib', '.bin'
        }
        
        # Licensing information patterns
        self.license_patterns = {
            'mit': r'(?i)MIT\s+License',
            'apache': r'(?i)Apache\s+License',
            'gpl': r'(?i)GNU\s+General\s+Public\s+License',
            'bsd': r'(?i)BSD\s+License',
            'copyright': r'(?i)Copyright\s+\(c\)\s+\d{4}',
        }
        
        # Maximum file size (in bytes) to process
        self.max_file_size = 1024 * 1024  # 1MB
        
    def is_file_allowed(self, filename: str, file_size: int) -> tuple[bool, str]:
        """Check if file should be processed based on extension and size"""
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext in self.blocked_extensions:
            return False, f"File type '{file_ext}' is not allowed for security reasons"
        
        if file_size > self.max_file_size:
            return False, f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
        
        return True, ""
    
    def detect_sensitive_data(self, content: str) -> Dict[str, List[str]]:
        """Detect sensitive data patterns in content"""
        findings = {}
        
        for category, patterns in self.sensitive_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                if found:
                    matches.extend(found)
            
            if matches:
                findings[category] = matches
        
        return findings
    
    def detect_inappropriate_content(self, content: str) -> List[str]:
        """Detect inappropriate or vulgar content"""
        findings = []
        
        for pattern in self.inappropriate_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            findings.extend(matches)
        
        return findings
    
    def detect_license_info(self, content: str) -> Dict[str, bool]:
        """Detect licensing information in content"""
        licenses = {}
        
        for license_type, pattern in self.license_patterns.items():
            licenses[license_type] = bool(re.search(pattern, content, re.MULTILINE | re.IGNORECASE))
        
        return licenses
    
    def redact_sensitive_content(self, content: str, findings: Dict[str, List[str]]) -> str:
        """Redact sensitive information from content"""
        redacted_content = content
        
        for category, patterns in self.sensitive_patterns.items():
            if category in findings:
                for pattern in patterns:
                    redacted_content = re.sub(
                        pattern, 
                        f"[REDACTED_{category.upper()}]", 
                        redacted_content, 
                        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
                    )
        
        return redacted_content
    
    def validate_and_filter_content(self, content: str, filename: str, file_size: int) -> Dict[str, Any]:
        """Main validation and filtering method"""
        # Check if file is allowed
        is_allowed, reason = self.is_file_allowed(filename, file_size)
        if not is_allowed:
            return {
                'allowed': False,
                'reason': reason,
                'content': None,
                'warnings': [],
                'license_info': {},
                'source_attribution': None
            }
        
        # Detect sensitive data
        sensitive_findings = self.detect_sensitive_data(content)
        
        # Detect inappropriate content
        inappropriate_findings = self.detect_inappropriate_content(content)
        
        # Detect license information
        license_info = self.detect_license_info(content)
        
        # Build warnings
        warnings = []
        if sensitive_findings:
            warnings.append(f"Sensitive data detected and redacted: {list(sensitive_findings.keys())}")
        
        if inappropriate_findings:
            warnings.append("Inappropriate content detected and filtered")
            return {
                'allowed': False,
                'reason': 'Content contains inappropriate material',
                'content': None,
                'warnings': warnings,
                'license_info': license_info,
                'source_attribution': None
            }
        
        # Redact sensitive content
        filtered_content = self.redact_sensitive_content(content, sensitive_findings)
        
        # Add license warning if no clear license found
        if not any(license_info.values()) and len(content) > 100:  # Only for substantial content
            warnings.append("No clear license information found - use with caution regarding copyright")
        
        return {
            'allowed': True,
            'reason': '',
            'content': filtered_content,
            'warnings': warnings,
            'license_info': license_info,
            'source_attribution': f"Source: GitHub repository content"
        }


