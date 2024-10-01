let chatOpened = false;

async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    addMessageToChatbox('You', userInput, 'user-message');
    document.getElementById('user-input').value = '';

    try {
        const response = await get_response(userInput)
        addMessageToChatbox('Assistant', response, 'bot-message')
    } catch (error) {
        addMessageToChatbox('Assistant', 'Error fetching response: ' + error.message, 'bot-message');
    }
}

async function get_response (userInput){
    const response = await fetch(`http://localhost:8000/api/getResponse?userInput=${userInput}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    });

    const data = await response.json();
    return data['response'];
}





function addMessageToChatbox(sender, message, messageType) {
    const chatbox = document.getElementById('chatbox');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${messageType}`;
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message.replace(/\n/g, '<br>')}`;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function toggleChat() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.classList.toggle('visible');

    if (!chatOpened) {
        addMessageToChatbox('Assistant', 'Welcome to Agentic Chatbot', 'bot-message');
        chatOpened = true;
    }
}

document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});
