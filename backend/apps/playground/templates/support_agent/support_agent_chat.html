{% extends 'base.html' %}
{% load static %}
{% load custom_markdown custom_filters %}

{% block title %}Order Support - #{{ order.id }}{% endblock %}

{% block on_page_css %}
<link rel="stylesheet" href="{% static 'support_agent/css/styles.css' %}">
{% endblock on_page_css %}

{% block content %}
<div class="container py-4">
    <!-- Order Header -->
    <div class="row mb-4">
        <div class="col">
            <h1 class="h3">Order #{{ order.id }}</h1>
            <p class="text-muted">Placed on {{ order.created|date:"F j, Y" }}</p>
        </div>
        <div class="col-auto">
            <span class="status-badge status-{{ order.status|lower }}">
                {{ order.get_status_display }}
            </span>
        </div>
    </div>

    <!-- Chat Interface -->
    <div class="card mb-4" id="chat-interface">
        <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="card-title mb-0">Customer Support Agent</h5>
            <small class="text-muted">Ask me anything about your order</small>
        </div>
        <div class="card-body">
            <!-- Chat Messages Container -->
            <div class="chat-messages" id="chat-messages">
                <!-- Messages will be inserted here -->
            </div>
            
            <!-- Chat Input -->
            <div class="chat-input">
                <div class="input-group">
                    <input type="text" class="form-control" id="user-input" 
                           placeholder="Ask about tracking, modifications, or any other order-related questions...">
                    <button class="btn btn-primary" id="send-message">
                        Send <i class="fas fa-paper-plane ms-1"></i>
                    </button>
                </div>
                <small class="form-text text-muted">
                    Example questions: "Where is my order?", "Can I change the quantity?", "What's the delivery status?"
                </small>
            </div>
        </div>
    </div>

    <!-- Loading Spinner -->
    <div class="loading-spinner" id="loading-spinner" style="display: none;">
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Processing your request...</p>
    </div>

    <!-- Confirmation Dialog -->
    <div class="confirmation-dialog" id="confirmation-dialog" style="display: none;">
        <div class="confirmation-content">
            <h5>Confirm Action</h5>
            <div id="confirmation-message"></div>
            <div class="confirmation-buttons mt-3">
                <button class="btn btn-primary" onclick="window.orderSupport.handleConfirmation(true)">Approve</button>
                <button class="btn btn-secondary" onclick="window.orderSupport.handleConfirmation(false)">Decline</button>
            </div>
            <div class="confirmation-reason mt-3" id="confirmation-reason" style="display: none;">
                <label for="decline-reason" class="form-label">Please provide a reason:</label>
                <textarea class="form-control" id="decline-reason" rows="2"></textarea>
                <button class="btn btn-danger mt-2" onclick="window.orderSupport.declineWithReason()">Submit Reason</button>
            </div>
        </div>
    </div>

    <!-- Toast Container for Notifications -->
    <div class="position-fixed bottom-0 end-0 p-3" style="z-index: 1050">
        <div class="toast-container"></div>
    </div>
</div>
{% endblock %}

{% block on_page_js %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/2.3.3/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
// Configure marked for Markdown parsing
marked.setOptions({
    breaks: true,
    gfm: true,
    sanitize: false,
    headerIds: false
});

class SupportAgent {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        this._conversationUUID = null;
        this.orderID = '{{ order.id }}';
        this.currentToolCalls = null;
        this.pendingMessages = [];
        this.MAX_RECONNECT_ATTEMPTS = 5;
        this.RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000];
        
        this.initializeEventListeners();
        this.connectWebSocket();
    }

    initializeEventListeners() {
        // Chat input listeners
        document.getElementById('send-message').addEventListener('click', () => this.handleUserInput());
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleUserInput();
            }
        });
    }

    connectWebSocket() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
        if (this.isReconnecting) return;

        try {
            const wsScheme = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const wsPath = `${wsScheme}${window.location.host}/ws/support_agent/${this.conversationUUID}/`;
            
            this.ws = new WebSocket(wsPath);
            this.setupWebSocketHandlers();
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            this.handleReconnection();
        }
    }

    setupWebSocketHandlers() {
        this.ws.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.isReconnecting = false;
            
            while (this.pendingMessages.length > 0) {
                const message = this.pendingMessages.shift();
                this.sendMessage(message);
            }
        };

        this.ws.onclose = (e) => {
            console.log('WebSocket closed:', e.code, e.reason);
            this.handleReconnection();
        };

        this.ws.onerror = (e) => {
            console.error('WebSocket error:', e);
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.close();
            }
        };

        this.ws.onmessage = (event) => this.handleWebSocketMessage(event);
    }

    handleReconnection() {
        if (this.isReconnecting || this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
            if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
                this.showToast('Error', 'Connection lost. Please refresh the page.');
            }
            return;
        }

        this.isReconnecting = true;
        const delay = this.RECONNECT_DELAYS[this.reconnectAttempts] || 
                     this.RECONNECT_DELAYS[this.RECONNECT_DELAYS.length - 1];
        
        setTimeout(() => {
            this.reconnectAttempts++;
            this.isReconnecting = false;
            this.connectWebSocket();
        }, delay);
    }

    get conversationUUID() {
        if (!this._conversationUUID) {
            const serverProvidedUUID = '{{ conversation_id }}';
            this._conversationUUID = serverProvidedUUID || this.generateUUID();
        }
        return this._conversationUUID;
    }

    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    sendMessage(messageData) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.pendingMessages.push(messageData);
            this.connectWebSocket();
            return;
        }

        try {
            messageData.uuid = this.conversationUUID;
            this.ws.send(JSON.stringify(messageData));
        } catch (error) {
            console.error('Error sending message:', error);
            this.handleReconnection();
        }
    }

    handleUserInput() {
        const inputElement = document.getElementById('user-input');
        const message = inputElement.value.trim();
        
        if (!message) return;
        
        // Show loading spinner when sending message
        document.getElementById('loading-spinner').style.display = 'block';
        
        // Append user message locally
        this.appendMessage(message, true);
        
        // Clear input
        inputElement.value = '';
        
        const messageData = {
            type: 'message',
            message: message,
            order_id: this.orderID
        };
        this.sendMessage(messageData);
    }

    handleWebSocketMessage(event) {
        const data = JSON.parse(event.data);
        const loadingSpinner = document.getElementById('loading-spinner');
        
        console.log("Received WebSocket message:", data);
        
        try {
            switch(data.type) {
                case 'welcome':
                    this.appendMessage(data.message, false, 'system');
                    break;
                
                case 'agent_response':
                    if (data.message) {
                        this.appendMessage(data.message, false);
                        }
                    if (data.tool_call) {
                        console.log("Tool being called:", data.tool_call);
                    }
                    break

                case 'confirmation_required':
                    console.log("Showing confirmation dialog:", data);
                    this.showConfirmationDialog(data.action, data.tool_calls);
                    break;

                case 'operation_complete':
                    this.handleOperationComplete(data);
                    break;

                case 'processing_complete':
                    // Hide spinner when processing is complete
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    break;

                case 'error':
                    this.showToast('Error', data.message, 'error');
                    // Ensure spinner is hidden on error
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    break;

                case 'context_update':
                    // Handle any context updates from the backend
                    if (data.order_status) {
                        this.updateOrderStatus(data.order_status);
                    }
                    break;
            }

            // Always scroll to bottom after new messages
            this.scrollToBottom();
            
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
            // Ensure spinner is hidden on error
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            this.showToast('Error', 'An error occurred while processing the message', 'error');
        }
    }

    handleOperationComplete(data) {
        document.getElementById('confirmation-dialog').style.display = 'none';
        document.querySelector('.confirmation-overlay')?.remove();
        
        if (data.message) {
            this.appendMessage(data.message, false, 'tool-response');
        }
        
        if (data.update_elements) {
            this.updateUIElements(data.update_elements);
        }
        
        if (data.completion_message) {
            this.showToast('Success', data.completion_message, 'success');
        }
    }

    showConfirmationDialog(action, toolCalls) {
        document.querySelector('.confirmation-overlay')?.remove();
        
        const overlay = document.createElement('div');
        overlay.className = 'confirmation-overlay';
        document.body.appendChild(overlay);

        const dialog = document.getElementById('confirmation-dialog');
        const message = document.getElementById('confirmation-message');
        
        this.currentToolCalls = toolCalls;

        message.innerHTML = DOMPurify.sanitize(`
            <p><strong>Please Confirm Action:</strong></p>
            <p>${marked.parse(action)}</p>
            ${toolCalls && toolCalls.length > 0 ? `
                <p><strong>Changes to be made:</strong></p>
                <ul>
                    ${toolCalls.map(tc => `<li>${this.formatToolCall(tc)}</li>`).join('')}
                </ul>
            ` : ''}
            <div class="alert alert-info">
                Please confirm if you want to proceed with these changes.
            </div>
        `);

        dialog.style.display = 'block';
        document.getElementById('confirmation-reason').style.display = 'none';
    }

    formatToolCall(toolCall) {
        try {
            const args = typeof toolCall.function.arguments === 'string' 
                ? JSON.parse(toolCall.function.arguments) 
                : toolCall.function.arguments;

            switch(toolCall.function.name) {
                case 'modify_order_quantity':
                    return `Change quantity of product (ID: ${args.product_id}) to ${args.new_quantity}`;
                case 'cancel_order':
                    return `Cancel order ${args.order_id}`;
                case 'track_order':
                    return `Track order ${args.order_id}`;
                default:
                    return `${toolCall.function.name}: ${JSON.stringify(args)}`;
            }
        } catch (error) {
            console.error('Error formatting tool call:', error);
            return 'Error formatting tool details';
        }
    }

    handleConfirmation(approved) {
        const dialog = document.getElementById('confirmation-dialog');
        const loadingSpinner = document.getElementById('loading-spinner');
        const reasonElement = document.getElementById('confirmation-reason');
        
        if (!approved) {
            reasonElement.style.display = 'block';
            return;
        }
        
        dialog.style.display = 'none';
        document.querySelector('.confirmation-overlay')?.remove();
        loadingSpinner.style.display = 'block';
        
        const messageData = {
            type: 'confirmation',
            approved: true,
            tool_calls: this.currentToolCalls
        };
        
        this.sendMessage(messageData);
        this.currentToolCalls = null;
    }

    declineWithReason() {
        const reason = document.getElementById('decline-reason').value.trim() || 'Action declined by user';
        const messageData = {
            type: 'confirmation',
            approved: false,
            tool_calls: this.currentToolCalls,
            reason: reason
        };
        
        document.getElementById('confirmation-dialog').style.display = 'none';
        document.querySelector('.confirmation-overlay')?.remove();
        document.getElementById('loading-spinner').style.display = 'block';
        
        this.sendMessage(messageData);
        this.currentToolCalls = null;
    }

    appendMessage(content, isUser = false, messageType = 'regular') {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');

        let messageClasses = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        if (messageType === 'tool-response') {
            messageClasses += ' tool-response';
        } else if (messageType === 'system') {
            messageClasses += ' system-message';
        }
        
        messageDiv.className = messageClasses;
        
        // Clean and parse markdown only for assistant messages
        const messageContent = isUser ? content : DOMPurify.sanitize(marked.parse(content));
        messageDiv.innerHTML = `
            ${messageContent}
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    updateUIElements(updates) {
        updates.forEach(update => {
            const element = document.querySelector(update.selector);
            if (element) {
                element.outerHTML = DOMPurify.sanitize(update.html);
            }
        });
    }

    updateOrderStatus(status) {
        const statusBadge = document.querySelector('.status-badge');
        if (statusBadge) {
            statusBadge.className = `status-badge status-${status.toLowerCase()}`;
            statusBadge.textContent = status;
        }
    }

    showToast(title, message, type = 'info') {
        const toastContainer = document.querySelector('.toast-container');
        const toast = document.createElement('div');
        toast.className = `toast show bg-${type === 'error' ? 'danger' : type} text-white`;
        toast.setAttribute('role', 'alert');
        
        toast.innerHTML = `
            <div class="toast-header bg-${type === 'error' ? 'danger' : type} text-white">
                <strong class="me-auto">${title}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;
        
        toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Initialize SupportAgent when document is ready
document.addEventListener('DOMContentLoaded', () => {
    window.orderSupport = new SupportAgent();
});
</script>
{% endblock %}