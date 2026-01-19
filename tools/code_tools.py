"""Code generation and analysis tools."""

import re
from typing import Optional, List, Tuple, Union

from models.schemas import Language


class CodeTools:
    """Utility tools for code processing."""
    
    # Language file extensions
    EXTENSIONS = {
        Language.PYTHON: ".py",
        Language.JAVASCRIPT: ".js",
        Language.TYPESCRIPT: ".ts",
        Language.JAVA: ".java",
        Language.CPP: ".cpp",
        Language.GO: ".go",
        Language.RUST: ".rs",
        Language.SQL: ".sql",
        Language.HTML: ".html",
        Language.CSS: ".css",
    }
    
    # Language syntax highlighting names
    SYNTAX_NAMES = {
        Language.PYTHON: "python",
        Language.JAVASCRIPT: "javascript",
        Language.TYPESCRIPT: "typescript",
        Language.JAVA: "java",
        Language.CPP: "cpp",
        Language.GO: "go",
        Language.RUST: "rust",
        Language.SQL: "sql",
        Language.HTML: "html",
        Language.CSS: "css",
    }
    
    @staticmethod
    def detect_language(code: str) -> Language:
        """Detect programming language from code content.
        
        Args:
            code: Code string to analyze.
            
        Returns:
            Detected Language enum value.
        """
        code_lower = code.lower()
        
        # Python indicators
        if any(x in code for x in ['def ', 'import ', 'from ', 'class ', 'if __name__']):
            if 'self' in code or 'def ' in code:
                return Language.PYTHON
        
        # TypeScript indicators
        if any(x in code for x in [': string', ': number', ': boolean', 'interface ', ': void']):
            return Language.TYPESCRIPT
        
        # JavaScript indicators
        if any(x in code for x in ['const ', 'let ', 'function ', '=>', 'require(']):
            return Language.JAVASCRIPT
        
        # Java indicators
        if any(x in code for x in ['public class', 'public static void', 'System.out']):
            return Language.JAVA
        
        # C++ indicators
        if any(x in code for x in ['#include', 'std::', 'cout', 'cin', 'int main()']):
            return Language.CPP
        
        # Go indicators
        if any(x in code for x in ['func ', 'package ', 'import (', 'fmt.']):
            return Language.GO
        
        # Rust indicators
        if any(x in code for x in ['fn ', 'let mut', 'impl ', '-> ', 'pub fn']):
            return Language.RUST
        
        # SQL indicators
        if any(x in code_lower for x in ['select ', 'insert ', 'update ', 'delete ', 'create table']):
            return Language.SQL
        
        # HTML indicators
        if any(x in code_lower for x in ['<!doctype', '<html', '<head>', '<body>']):
            return Language.HTML
        
        # CSS indicators
        if re.search(r'[\w-]+\s*:\s*[\w-]+\s*;', code) and '{' in code:
            if not any(x in code for x in ['function', 'const ', 'let ']):
                return Language.CSS
        
        return Language.OTHER
    
    @staticmethod
    def get_extension(language: Language) -> str:
        """Get file extension for a language.
        
        Args:
            language: Programming language.
            
        Returns:
            File extension with dot.
        """
        return CodeTools.EXTENSIONS.get(language, ".txt")
    
    @staticmethod
    def get_syntax_name(language: Language) -> str:
        """Get syntax highlighting name for a language.
        
        Args:
            language: Programming language.
            
        Returns:
            Syntax highlighting identifier.
        """
        return CodeTools.SYNTAX_NAMES.get(language, "text")
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown text.
        
        Args:
            text: Markdown text containing code blocks.
            
        Returns:
            List of (language, code) tuples.
        """
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang or "text", code.strip()) for lang, code in matches]
    
    @staticmethod
    def format_code_block(code: str, language: Union[Language, str]) -> str:
        """Format code as markdown code block.
        
        Args:
            code: Code string.
            language: Programming language or syntax name.
            
        Returns:
            Markdown formatted code block.
        """
        if isinstance(language, Language):
            syntax = CodeTools.get_syntax_name(language)
        else:
            syntax = language
        
        return f"```{syntax}\n{code}\n```"
    
    @staticmethod
    def count_lines(code: str) -> int:
        """Count lines in code.
        
        Args:
            code: Code string.
            
        Returns:
            Number of lines.
        """
        return len(code.splitlines())
    
    @staticmethod
    def suggest_filename(description: str, language: Language) -> str:
        """Suggest a filename based on description.
        
        Args:
            description: Description of the code.
            language: Programming language.
            
        Returns:
            Suggested filename.
        """
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        
        # Filter common words
        stop_words = {'a', 'an', 'the', 'to', 'for', 'of', 'in', 'on', 'with', 'that', 'this', 'is', 'are'}
        words = [w for w in words if w not in stop_words][:3]
        
        if words:
            name = '_'.join(words)
        else:
            name = "generated_code"
        
        extension = CodeTools.get_extension(language)
        return f"{name}{extension}"
    
    @staticmethod
    def validate_syntax(code: str, language: Language) -> Tuple[bool, Optional[str]]:
        """Basic syntax validation (mainly for Python).
        
        Args:
            code: Code to validate.
            language: Programming language.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        if language == Language.PYTHON:
            try:
                compile(code, "<string>", "exec")
                return True, None
            except SyntaxError as e:
                return False, f"Syntax error at line {e.lineno}: {e.msg}"
        
        # For other languages, just do basic bracket matching
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    return False, "Mismatched brackets"
        
        if stack:
            return False, "Unclosed brackets"
        
        return True, None
