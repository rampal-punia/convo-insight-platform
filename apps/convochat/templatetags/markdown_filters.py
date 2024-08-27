from django import template
from markdown import markdown
import bleach
import re
from markdown.extensions.nl2br import Nl2BrExtension

register = template.Library()


@register.filter(name='markdown_to_html')
def markdown_to_html(value):
    # Remove the </s> token
    value = value.replace('AI:', '')
    value = value.replace('</s>', '')

    # Normalize line endings and remove excessive blank lines
    value = re.sub(r'\r\n|\r', '\n', value)
    value = re.sub(r'\n{3,}', '\n\n', value)

    # Convert markdown to HTML with nl2br extension
    html = markdown(value, extensions=[
                    'fenced_code', 'tables', Nl2BrExtension()])

    # Sanitize the HTML to prevent XSS attacks
    allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'strong', 'em',
                    'ul', 'ol', 'li', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td']
    allowed_attributes = {'*': ['class']}
    clean_html = bleach.clean(html, tags=allowed_tags,
                              attributes=allowed_attributes)

    # Replace &lt; and &gt; within <pre> tags
    clean_html = re.sub(r'<pre><code>(.*?)</code></pre>', lambda m: '<pre><code>' + m.group(
        1).replace('&lt;', '<').replace('&gt;', '>') + '</code></pre>', clean_html, flags=re.DOTALL)

    # Add custom CSS for spacing and list styles
    custom_css = """
    <style>
    .message-content p { margin-bottom: 0.5em; }
    .message-content h1, .message-content h2, .message-content h3, 
    .message-content h4, .message-content h5, .message-content h6 { 
        margin-top: 0.5em; 
        margin-bottom: 0.5em; 
    }
    .message-content pre { 
        margin-top: 0.5em; 
        margin-bottom: 0.5em;
        white-space: pre-wrap;
    }
    .message-content ul, .message-content ol { 
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        padding-left: 1.5em;
    }
    .message-content li {
        margin-bottom: 0.3em;
    }
    .message-content li > p {
        margin-top: 0.2em;
        margin-bottom: 0.2em;
    }
    </style>
    """

    return custom_css + clean_html
