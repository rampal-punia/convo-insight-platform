from django import template
from markdown import markdown
import bleach
from markdown.extensions.nl2br import Nl2BrExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from django.utils.safestring import mark_safe
from django.template.defaultfilters import stringfilter

register = template.Library()

# Allowed tags and attributes for sanitization
ALLOWED_TAGS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'br', 'strong', 'em', 'a', 'code', 'pre',
    'ul', 'ol', 'li', 'table', 'thead', 'tbody',
    'tr', 'th', 'td', 'blockquote', 'hr'
]

ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title', 'rel'],
    'code': ['class'],  # For syntax highlighting
    'pre': ['class'],   # For wrapping code blocks
}

MARKDOWN_EXTENSIONS = [
    FencedCodeExtension(),
    TableExtension(),
    Nl2BrExtension(),
    'markdown.extensions.sane_lists',
]


@register.filter(name='markdown_to_html')
@stringfilter
def markdown_to_html(value):
    """
    Converts Markdown text to sanitized HTML with proper code block rendering.
    """
    try:
        # Normalize line breaks to prevent Markdown processing issues
        value = value.replace('\r\n', '\n').replace('\r', '\n')

        # Convert Markdown to HTML
        html = markdown(
            value,
            extensions=MARKDOWN_EXTENSIONS
        )

        # Sanitize the HTML to remove unsafe tags/attributes
        sanitized_html = bleach.clean(
            html,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True
        )

        # Enhance code block rendering
        sanitized_html = fix_all_code_blocks(sanitized_html)

        # Wrap in a container div for styling
        final_html = f'<div class="message-content">{sanitized_html}</div>'

        return mark_safe(final_html)
    except Exception as e:
        # Handle errors gracefully
        return mark_safe(f'<p>Error processing markdown: {str(e)}</p>')


def fix_all_code_blocks(html):
    """
    Enhance code block rendering for both inline and block-level code.
    """
    import re

    # Handle fenced code blocks (<pre><code>)
    block_pattern = r'<pre><code(?: class="language-([^"]+)")?>(.*?)</code></pre>'
    inline_pattern = r'<code(?: class="language-([^"]+)")?>(.*?)</code>'

    def replace_block_code(match):
        # Default to 'text' if no language is specified
        language = match.group(1) or 'text'
        code_content = match.group(2)

        # Decode HTML entities for proper rendering
        code_content = (
            code_content.replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&amp;', '&')
            .replace('&quot;', '"')
        )

        # Return formatted block code
        return f'<pre class="language-{language}"><code class="language-{language}">{code_content}</code></pre>'

    def replace_inline_code(match):
        # Inline code usually doesn't have a language
        language = match.group(1) or 'text'
        code_content = match.group(2)

        # Decode HTML entities
        code_content = (
            code_content.replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&amp;', '&')
            .replace('&quot;', '"')
        )

        # Return formatted inline code
        return f'<code class="language-{language}">{code_content}</code>'

    # Apply fixes for block-level code
    html = re.sub(block_pattern, replace_block_code, html, flags=re.DOTALL)

    # Apply fixes for inline code
    html = re.sub(inline_pattern, replace_inline_code, html, flags=re.DOTALL)

    return html
