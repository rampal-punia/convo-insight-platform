from django import template
from markdown2 import markdown
import bleach

register = template.Library()


@register.filter(name='markdown_to_html')
def markdown_to_html(value):
    # Convert markdown to html
    html = markdown(
        value, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])

    # Sanitize the HTML to prevent XSS attacks
    allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'strong', 'em',
                    'ul', 'ol', 'li', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td']
    allowed_attributes = {'*': ['class']}
    clean_html = bleach.clean(html, tags=allowed_tags,
                              attributes=allowed_attributes)

    return clean_html
