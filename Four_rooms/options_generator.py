# module to generate options
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

options = []

# defining options for room 0
option_list = []
# option for entering hallway 0
option1 = [RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
           RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
           RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,
           RIGHT, RIGHT, RIGHT, RIGHT, UP,
           RIGHT, RIGHT, RIGHT, RIGHT, UP]
options.append(option1)
# option_list.append(option1)
# option for entering hallway 3
option2 = [DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           RIGHT, DOWN, LEFT, LEFT, LEFT]
options.append(option2)
# option_list.append(option2)
# options.append(option_list)

# defining room 1 options
option_list = []
# option for entering hallway 0
option1 = [DOWN, LEFT, LEFT, LEFT, LEFT,
           DOWN, LEFT, LEFT, LEFT, LEFT,
           LEFT, LEFT, LEFT, LEFT, LEFT,
           UP, LEFT, LEFT, LEFT, LEFT,
           UP, LEFT, LEFT, LEFT, LEFT,
           UP, LEFT, LEFT, LEFT, LEFT]
options.append(option1)
# option_list.append(option1)
# option for entering hallway 1
option2 = [DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           DOWN, DOWN, DOWN, DOWN, DOWN,
           RIGHT, RIGHT, DOWN, LEFT, LEFT]
# option_list.append(option2)
# options.append(option_list)
options.append(option2)
# defining room 2 options
option_list = []
# option for entering hallway 1
option1 = [RIGHT, RIGHT, UP, LEFT, LEFT,
           UP, UP, UP, UP, UP,
           UP, UP, UP, UP, UP,
           UP, UP, UP, UP, UP]
options.append(option1)
# option_list.append(option1)
# option for entering hallway 2
option2 = [DOWN, LEFT, LEFT, LEFT, LEFT,
           DOWN, LEFT, LEFT, LEFT, LEFT,
           LEFT, LEFT, LEFT, LEFT, LEFT,
           UP, LEFT, LEFT, LEFT, LEFT]
# option_list.append(option2)
# options.append(option_list)
options.append(option2)
# defining room 3 options
option_list = []
# option for entering hallway 2
option1 = [RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
           RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
           RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
           RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,
           RIGHT, RIGHT, RIGHT, RIGHT, UP]
options.append(option1)
# option_list.append(option1)
# option for entering hallway 3
option2 = [RIGHT, UP, LEFT, LEFT, LEFT,
           UP, UP, UP, UP, UP,
           UP, UP, UP, UP, UP,
           UP, UP, UP, UP, UP,
           UP, UP, UP, UP, UP]
# option_list.append(option2)
# options.append(option_list)
options.append(option2)
options_encoded = []  # options_encoded[i][j] = action in option i for encoded index j
hmm = [103, 25, 56, 25, 77, 56, 103, 77]  # option i includes action for hallway with encoded index option[i]
hmm_action = [UP, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN, LEFT]  # action for hallway with encoded index option[i] in i
option_starts = [0, 0, 26, 26, 57, 57, 78, 78]  # start index for option i
terminal_hallways = [25, 103, 25, 56, 56, 77, 77, 103]  # target hallway for option i

# initializing options_encoded
for i in range(len(options)):
    curr = []
    for j in range(0, 104):
        curr.append(0)
    options_encoded.append(curr)

for i in range(len(options)):
    start = option_starts[i]
    for j in range(len(options[i])):
        options_encoded[i][start+j] = options[i][j]
    options_encoded[i][hmm[i]] = hmm_action[i]  # assigning action for non-target hallway state in option i


def get_options():
    return options_encoded


def get_terminal_hallways():
    return terminal_hallways
