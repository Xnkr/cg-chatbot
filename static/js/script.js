(function () {

    var chat = {
        messageToSend: '',
        messageResponses: [
            "I can't find my brain. Get my creator! Quick!",
            "Oops! Something went wrong with my brain"
        ],
        init: function () {
            this.cacheDOM();
            this.bindEvents();
            this.render();
        },
        cacheDOM: function () {
            this.$chatHistory = $('.chat-history');
            this.$button = $('button');
            this.$textarea = $('#message-to-send');
            this.$chatHistoryList = this.$chatHistory.find('ul');
        },
        bindEvents: function () {
            this.$button.on('click', this.addMessage.bind(this));
            this.$textarea.on('keyup', this.addMessageEnter.bind(this));
        },
        render: function () {
            this.scrollToBottom();
            if (this.messageToSend.trim() !== '') {
                var template = Handlebars.compile($("#message-template").html());
                var context = {
                    messageOutput: this.messageToSend,
                    time: this.getCurrentTime()
                };

                this.$chatHistoryList.find('#wait-init').remove();

                this.$chatHistoryList.append(template(context));
                this.scrollToBottom();
                this.$textarea.val('');

                var loadTemplate = Handlebars.compile($("#message-wait").html());
                var loadContext = {
                    time: this.getCurrentTime()
                };

                this.$chatHistoryList.append(loadTemplate(loadContext));
                this.scrollToBottom();

                var templateResponse = Handlebars.compile($("#message-response-template").html());
                $.ajax({
                    url: 'http://127.0.0.1:5000/chatbot',
                    type: 'post',
                    dataType: 'json',
                    contentType: 'application/json',
                    success: (function (data) {
                        // responses
                        var contextResponse = {
                            response: data.prediction,
                            time: this.getCurrentTime()
                        };
                        this.$chatHistoryList.find('#wait').remove();
                        this.$chatHistoryList.append(templateResponse(contextResponse));
                        this.scrollToBottom();
                    }).bind(this),
                    failure: (function(data){
                      // responses
                        var contextResponse = {
                            response: this.getRandomItem(this.messageResponses),
                            time: this.getCurrentTime()
                        };
                        this.$chatHistoryList.find('#wait').remove();
                        this.$chatHistoryList.append(templateResponse(contextResponse));
                        this.scrollToBottom();
                    }).bind(this),
                    data: JSON.stringify({'query': this.messageToSend.trim()})
                });

            }

        },

        addMessage: function () {
            this.messageToSend = this.$textarea.val()
            this.render();
        },
        addMessageEnter: function (event) {
            // enter was pressed
            if (event.keyCode === 13) {
                this.addMessage();
            }
        },
        scrollToBottom: function () {
            this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
        },
        getCurrentTime: function () {
            return new Date().toLocaleTimeString().replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
        },
        getRandomItem: function (arr) {
            return arr[Math.floor(Math.random() * arr.length)];
        }

    };

    chat.init();

})();
