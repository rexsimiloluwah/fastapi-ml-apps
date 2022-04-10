class Chatbot{
    constructor() {
        this.elements = {
            openChatboxButton: document.querySelector(".chatbox__button"),
            chatBox: document.querySelector(".chatbox__support"),
            sendChatButton: document.querySelector(".send__button")
        }
        this.displayState = false; 
        this.messages = [];
    }

    display(){
        const {openChatboxButton, chatBox, sendChatButton} = this.elements; 

        openChatboxButton.addEventListener("click", () => {
            this.toggleChatbox()
        })

        sendChatButton.addEventListener("click", () => {
            this.sendMessage()
        })

        const chatInput = chatBox.querySelector("input")
        chatInput.addEventListener("keyup", (e) => {
            if(e.key === "Enter"){
                this.sendMessage()
            }
        })
    }

    toggleChatbox(){
        console.log("Toggled chatbox.")
        if(!this.displayState){
            this.elements.chatBox.classList.add("chatbox--active");
        }else{
            this.elements.chatBox.classList.remove("chatbox--active");
        }
        this.displayState = !this.displayState;
    }

    sendMessage(){
        const chatInput = this.elements.chatBox.querySelector("input");
        let chatText = chatInput.value; 
        let msgSender = {id:"sender",value: chatText}
        this.messages.push(msgSender)
        fetch("/chat/predict", {
            method: "POST",
            body: JSON.stringify({message:chatText}),
            mode: "cors", 
            headers: {
                'Content-Type':'application/json'
            }
        })
        .then(r => r.json())
        .then(r => {
            let {response} = r; 
            const msgBot = {id:"bot", value: response};
            this.messages.push(msgBot);
            this.updateChatbox();
            chatInput.value = "";
        })
        .catch(error => {
            console.error('Error: ',error);
            this.updateChatbox();
            chatInput.value = "";
        })
    }

    updateChatbox(){
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.id === "sender")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.value + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.value + '</div>'
            }
          });

        const chatmessage = this.elements.chatBox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

let chatbot = new Chatbot();
chatbot.display();