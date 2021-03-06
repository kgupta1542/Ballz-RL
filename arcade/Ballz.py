import arcade
import numpy as np
import random

# Game Config Constants -----------------------------------------------------------------
SCREEN_WIDTH = 360
SCREEN_HEIGHT = 480
SCREEN_TITLE = "Ballz v2"

# Gameplay Constants --------------------------------------------------------------------
# Ball Constants
BALL_INIT_HEIGHT = 30
BALL_RADIUS = 5
BALL_COLL_PADDING = 1
BALl_COLL_RADIUS = BALL_RADIUS + BALL_COLL_PADDING
BALL_MAX_SPEED = 300
BALL_HIT_POINTS = 1

# Block Constants
BLOCK_SIZE = 36
BLOCK_PADDING = 2
BLOCK_OFFSET = BLOCK_SIZE / 2 + 1
BLOCK_SIZE_WITH_PADDING = 2*BLOCK_PADDING + BLOCK_SIZE
BLOCK_IS_EMPTY_CHANCE = 0.75
BLOCK_MAX_VALUE = 2

# Text Constants
GREEN = (0, 255, 0)
FONT_SIZE = 18

# Sizing Calculations ----------------------------------------------------------------------
INIT_EMPTY_ROWS = 4
NUM_BLOCKS_X = round((SCREEN_WIDTH - 2)/BLOCK_SIZE_WITH_PADDING)
MAX_BLOCKS_Y = round((SCREEN_HEIGHT - BALL_INIT_HEIGHT - 2)/BLOCK_SIZE_WITH_PADDING)
NUM_BLOCKS_Y = MAX_BLOCKS_Y - INIT_EMPTY_ROWS

# Reinforcement Learning Constants -----------------------------------------------------------
TOTAL_NUM_EPISODES = 100
EPISODES_PER_STRATEGY = 10

# Custom Game Class ----------------------------------------------------------------------
class Ballz(arcade.Window):
    # Define sprites and game window
    def __init__(self):
        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.csscolor.BLACK)

        # Create main SpriteLists
        self.block_list = arcade.SpriteList(use_spatial_hash=True)
        self.ball_list = arcade.SpriteList()

        # Initial 2D Array representing block values
        self.blockArray = np.zeros((MAX_BLOCKS_Y, NUM_BLOCKS_X), np.int8)

        # Setup for reward function
        self.reward = 0
        self.totalBlockHitsPossible = 0
        self.blockHitsThisEpisode = 0

        # Setup for RL training
        self.strategies = []
        self.avgRewards = []

        self.strategy = []
        self.bestStrategy = []
        self.bestAvgReward = 0
        self.currAvgReward = 0

        # Setup for RL training
        self.currEpisode = 0
        self.numEpisodes = TOTAL_NUM_EPISODES
        self.angle = 45

        # Reference to single ball
        self.player = arcade.SpriteCircle(BALL_RADIUS, arcade.color.WHITE)
        self.player.center_x = SCREEN_WIDTH / 2
        self.player.center_y = BALL_INIT_HEIGHT
        self.hasBallExited = False  # Flag to make sure ball exits dead zone before setting up next episode

    def reset(self):
        # Create main SpriteLists
        self.block_list = arcade.SpriteList(use_spatial_hash=True)

        # Setup for reward function
        self.totalBlockHitsPossible = 0
        self.blockHitsThisEpisode = 0

        # Initial 2D Array representing block values
        self.blockArray = np.zeros((MAX_BLOCKS_Y, NUM_BLOCKS_X), np.int8)
        self.createBlocksAndArray()

        # Reference to single ball
        self.player.center_x = SCREEN_HEIGHT / 2
        self.player.center_y = BALL_INIT_HEIGHT
        self.hasBallExited = False  # Flag to make sure ball exits dead zone before setting up next episode

        self.currEpisode += 1

    # Set initial ball angle and create blocks
    def setup(self):
        self.ball_list.append(self.player)
        self.createBlocksAndArray()

        self.updateAgentStrategy()
        self.chooseNextAngle()

        self.fireBall()

    # Randomly creates blocks
    def createBlocksAndArray(self):
        for x in range(NUM_BLOCKS_X):
            xPos = BLOCK_SIZE_WITH_PADDING * x + BLOCK_OFFSET
            for y in range(NUM_BLOCKS_Y):
                yPos = SCREEN_HEIGHT - BLOCK_OFFSET - BLOCK_SIZE_WITH_PADDING * y

                currVal = self.chooseBlockValue(BLOCK_IS_EMPTY_CHANCE, BLOCK_MAX_VALUE)
                self.totalBlockHitsPossible += currVal
                self.blockArray[y][x] = currVal

                if currVal > 0:
                    # Create block object
                    block = arcade.SpriteSolidColor(BLOCK_SIZE, BLOCK_SIZE, GREEN)
                    block.center_x = xPos
                    block.center_y = yPos
                    self.block_list.append(block)

    # Generate values for block
    def chooseBlockValue(self, emptyOdds, maxVal):
        if random.random() > emptyOdds:
            return random.randint(1, maxVal)
        return 0

    # Sets up game for next episode
    def setUpNextEpisode(self):
        self.moveBlocksDown()
        self.createNewTopRowBlocks()
        self.removeEmptyRowsFromBlockArray()
        self.setAllBlockLabels()

    # Moves all blocks on screen down
    def moveBlocksDown(self):
        for block in self.block_list:
            block.center_y -= BLOCK_SIZE_WITH_PADDING

    # Remove empty rows from block array
    def removeEmptyRowsFromBlockArray(self):
        emptyRow = np.zeros(NUM_BLOCKS_X, np.int8)

        while np.array_equal(self.blockArray[len(self.blockArray) - 1], emptyRow):
            if len(self.blockArray) == MAX_BLOCKS_Y:
                break

            self.blockArray = np.delete(self.blockArray, len(self.blockArray) - 1, 0)

    # Check if there are too many rows blocks
    def checkGameOver(self):
        if not np.array_equal(self.blockArray[MAX_BLOCKS_Y - 1], np.zeros(NUM_BLOCKS_X, np.int8)):
            self.endGame()
            return True
        return False

    # Ends the game
    def endGame(self):
        print("Game over")
        arcade.finish_render()
        arcade.close_window()

    # Creates a new top row of blocks
    def createNewTopRowBlocks(self):
        newRow = np.zeros([1, NUM_BLOCKS_X], np.int8)

        # Create all block sprites
        for i in range(NUM_BLOCKS_X):
            currVal = self.chooseBlockValue(BLOCK_IS_EMPTY_CHANCE, BLOCK_MAX_VALUE)
            self.totalBlockHitsPossible += currVal
            newRow[0][i] = currVal

            if currVal > 0:
                block = arcade.SpriteSolidColor(BLOCK_SIZE, BLOCK_SIZE, GREEN)
                block.center_x = BLOCK_SIZE_WITH_PADDING * i + BLOCK_OFFSET
                block.center_y = SCREEN_HEIGHT - BLOCK_OFFSET
                self.block_list.append(block)

        # Add to blockArray
        self.blockArray = np.concatenate((newRow, self.blockArray), axis=0)

    # Find the corresponding index in the array for a box on display
    def getBlockIndexInArray(self, block):
        x = round((block.center_x - BLOCK_OFFSET) / BLOCK_SIZE_WITH_PADDING)
        y = round((SCREEN_HEIGHT - block.center_y - BLOCK_OFFSET)/BLOCK_SIZE_WITH_PADDING)

        return y, x

    # Overlays text labels on top of blocks in the display
    def setAllBlockLabels(self):
        for x in range(NUM_BLOCKS_X):
            xPos = BLOCK_SIZE_WITH_PADDING * x + BLOCK_OFFSET
            for y in range(len(self.blockArray)):
                yPos = SCREEN_HEIGHT - BLOCK_OFFSET - BLOCK_SIZE_WITH_PADDING * y

                val = str(self.blockArray[y][x])

                if val != "0":
                    self.drawBlockLabel(val, xPos, yPos)

    # Updates the text on a single block
    def updateBlockLabel(self, block):
        row, col = self.getBlockIndexInArray(block)
        self.blockHitsThisEpisode += min([BALL_HIT_POINTS, self.blockArray[row][col]])
        self.blockArray[row][col] -= min([BALL_HIT_POINTS, self.blockArray[row][col]])

        val = self.blockArray[row][col]
        # Remove block if the value is 0
        if val <= 0:
            block.remove_from_sprite_lists()
        else:
            self.drawBlockLabel(str(val), block.center_x, block.center_y)

    # Simplified code for drawing block value
    # Takes in game coordinates of block, not array coordinates
    def drawBlockLabel(self, val, x, y):
        arcade.draw_text(val, x - 6 * len(val), y - 11, arcade.color.BLACK, FONT_SIZE)

    # Chooses the angle of the next shot
    def chooseNextAngle(self):
        # Some range between 15 and 165
        self.angle = np.dot(self.strategy, self.getObservationSpace()) % 150 + 15

    # Sets the angle of the ball
    # Input is degrees, not radians
    def fireBall(self):
        radians = np.deg2rad(self.angle)
        self.player.change_x = BALL_MAX_SPEED * np.cos(radians)
        self.player.change_y = BALL_MAX_SPEED * np.sin(radians)

    # Update reward
    def updateReward(self):
        self.reward = self.blockHitsThisEpisode / self.totalBlockHitsPossible

        self.totalBlockHitsPossible -= self.blockHitsThisEpisode
        self.blockHitsThisEpisode = 0

    # Get observation space for RL agent
    def getObservationSpace(self):
        observationSpace = np.concatenate([self.blockArray.flatten(), [self.player.center_x]])
        return observationSpace

    # Update agent strategy
    def updateAgentStrategy(self):
        # Set up training for next strategy
        if self.currEpisode % EPISODES_PER_STRATEGY == 0:
            self.strategies.append(self.strategy)
            self.avgRewards.append(self.currAvgReward)

            # Update best strategy
            if self.currAvgReward > self.bestAvgReward:
                self.bestStrategy = self.strategy
                self.bestAvgReward = self.currAvgReward

            self.strategy = np.random.uniform(-1, 1, MAX_BLOCKS_Y * NUM_BLOCKS_X + 1)
            self.currAvgReward = 0
        else:
            self.currAvgReward += self.reward / EPISODES_PER_STRATEGY

    # Initial render of game
    def on_draw(self):
        arcade.start_render()
        self.block_list.draw()
        self.ball_list.draw()
        self.setAllBlockLabels()

    # Physics
    def on_update(self, delta_time: float):
        # Moving ball ---------------------------------------------------------------
        self.player.center_x += self.player.change_x * delta_time
        self.player.center_y += self.player.change_y * delta_time

        if self.player.center_y > BALL_INIT_HEIGHT:
            self.hasBallExited = True

        # Block collision -----------------------------------------------------------
        # Finds all blocks made contact with
        block_hit_list = arcade.check_for_collision_with_list(self.player, self.block_list)

        for block in block_hit_list:
            # Identify which side of the block was hit
            collision_list = (abs(block.top - self.player.center_y),
                              abs(block.bottom - self.player.center_y),
                              abs(block.right - self.player.center_x),
                              abs(block.left - self.player.center_x))

            index = np.argmax(collision_list)

            # Reverse speed of ball correspondingly
            if index == 0 or index == 1:
                self.player.change_y *= -1
            elif index == 2 or index == 3:
                self.player.change_x *= -1

            # Decrement the label of the hit block
            self.updateBlockLabel(block)

        # Wall Collision/Bounce
        if self.player.center_x <= BALl_COLL_RADIUS or self.player.center_x >= SCREEN_WIDTH - BALl_COLL_RADIUS:
            self.player.change_x *= -1
        if self.player.center_y >= SCREEN_HEIGHT - BALl_COLL_RADIUS:
            self.player.change_y *= -1
        # Stop when reached initial height
        elif self.player.center_y <= BALL_INIT_HEIGHT and self.hasBallExited:
            # Stop ball
            self.player.change_x = 0
            self.player.change_y = 0

            self.player.center_y = BALL_INIT_HEIGHT
            self.hasBallExited = False

            # Reinforcement Learning Setup
            self.updateReward()
            print(self.reward)
            # This marks the end of an "episode" ----------------------------------------------------

            # At this point, the reward from the previous episode and observation space
            # for the current episode are ready for gym

            # Print average rewards
            if self.currEpisode == self.numEpisodes:
                print(self.avgRewards)
                self.currEpisode += 1

            # Train agent
            if self.currEpisode < self.numEpisodes:
                self.reset()
                self.updateAgentStrategy()
                self.chooseNextAngle()
                self.fireBall()
            else:
                self.strategy = self.bestStrategy
                # Let agent play game after training
                if not self.checkGameOver():
                    self.setUpNextEpisode()

                    # Ideally at this point, the agent would choose a new angle by modifying the global
                    # variable and then shoot the ball again.

                    self.chooseNextAngle()

                    # Shoot the ball to start next episode
                    self.fireBall()


def main():
    window = Ballz()
    window.setup()
    arcade.run()

    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
