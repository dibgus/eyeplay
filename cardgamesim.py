#blackjack, so i'm not worrying about suits
import random
CARD_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
DEALER_POLICY = 17 #The value the dealer should try to get
def deck_value(deck):
    value = 0
    aces = 0
    for card in deck:
        if card == "Jack" or card == "Queen" or card == "King":
            value += 10
        elif card == "Ace":
            aces+= 1 #we need to see if we want a 1 or a 11
        else:
            value += int(card)
    for i in range(0, aces):
        if (value + 11) < 21 or (value + 11) == 21 and aces - i == 1:
            value += 11
        else:
            value += 1
    return value
class game_state:
    deck = []
    playerhand = []
    dealerhand = []
    hole = None
    def __init__(self):
        for i in range(1, 4):
            for types in CARD_LABELS:
                self.deck.append(types)
        self.deck = random.sample(self.deck, len(self.deck))  # shuffle deck
        self.playerhand.append(self.deck.pop())
        self.dealerhand.append(self.deck.pop())

    # plays a single round. Dealer follows a policy to hit 16
    # The player passes hit (0 or 1) dictated by the neural network results
    # returns 0 if game has ended, 1 if it keeps going
    def play_round(self, hit):
        if hit:
            self.playerhand.append(self.deck.pop())
            print "hit!"
        if deck_value(self.dealerhand) < 17:
            self.dealerhand.append(self.deck.pop())
        elif not hit or deck_value(self.playerhand) > 21:
            return 0
        return 1
    #game_end will pop a reward value for q-learning
    def game_end(self):
        playervalue = 0
        dealervalue = 0
        for i in range(0, len(self.playerhand)):
            playervalue += deck_value(self.playerhand)

        for i in range(0, len(self.dealerhand)):
            dealervalue += deck_value(self.dealerhand)
        if playervalue == 21:
            return 15
        elif playervalue > 21:
            return -11
        elif playervalue > dealervalue:
            return 10
        elif playervalue == dealervalue:
            return 0
        elif dealervalue > 21:
            return 10
        else:
            return -10
    def print_readable_state(self):
        playervalue = 0
        dealervalue = 0
        for i in range(0, len(self.playerhand)):
            playervalue += deck_value(self.playerhand)

        for i in range(0, len(self.dealerhand)):
            dealervalue += deck_value(self.dealerhand)
        print("Player's hand(%s): " % playervalue + ", ".join(self.playerhand))
        print("Dealer's hand(%s): " % dealervalue + ", ".join(self.dealerhand))
        print("Perceived reward: %s" % self.game_end())

import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(210, kernel_initializer='lecun_uniform', input_shape=(2,))) #input layer. todo come back to this
model.add(Activation('relu'))
#dropout
model.add(Dense(200, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
#dropout
model.add(Dense(2, kernel_initializer='lecun_uniform')) #output: 0 for potential hit reward, 1 for potential stand reward
model.add(Activation('linear'))
rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

def train(epochs=10000,gamma=0.9,epsilon=1):
    for i in range(epochs):
        game = game_state()
        ingame = 1
        while ingame:
            qvalue = model.predict(numpy.array([deck_value(game.playerhand), len(game.dealerhand)]).reshape(1,2), batch_size=1)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = numpy.argmax(qvalue)
            ingame = game.play_round(action)
            qnew = model.predict(numpy.array([deck_value(game.playerhand), len(game.dealerhand)]).reshape(1,2), batch_size=1)
            qmax = numpy.max(qnew)
            y = numpy.zeros((1,2))
            y[:] = qvalue[:]
            reward = game.game_end() if ingame == 0 else -1
            if reward == -1: #intermediary reward. I may remove this.
                update = (reward + (gamma * qmax))
            else: #terminal state -- completing game
                update = reward
            y[0][action] = update #target out
            print("Game %s" % (i,))
            model.fit(numpy.array([game.playerhand, len(game.dealerhand)]).reshape(1,2), y, batch_size=1, nb_epoch=1, verbose=1)
            if not ingame:
                game.print_readable_state()
            if epsilon > 0.1:
                epsilon -= (1/epochs)

train()