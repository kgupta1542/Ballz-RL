import arcade
import numpy as np
import random

# Game Config Constants
SCREEN_WIDTH = 360
SCREEN_HEIGHT = 480
SCREEN_TITLE = "Ballz v2"

# Gameplay Constants
BALL_INIT_HEIGHT = 30
BALL_RADIUS = 5
BALL_COLL_PADDING = 1
BALl_COLL_RADIUS = BALL_RADIUS + BALL_COLL_PADDING
MAX_SPEED = 300

PARTICLE_SCALE = 0.03
PARTICLE_INIT_COUNT = 3
PARTICLE_INTERVAL = 0.1

BLOCK_SIZE = 36
BLOCK_PADDING = 2
BLOCK_OFFSET = BLOCK_SIZE / 2 + 1
BLOCK_SIZE_WITH_PADDING = 2*BLOCK_PADDING + BLOCK_SIZE
INIT_EMPTY_ROWS = 4

GREEN = (0, 255, 0)
FONT_SIZE = 18

# Sizing calculations
NUM_BLOCKS_X = round((SCREEN_WIDTH - 2)/BLOCK_SIZE_WITH_PADDING)
MAX_BLOCKS_Y = round((SCREEN_HEIGHT - BALL_INIT_HEIGHT + BLOCK_SIZE/2 - 2)/BLOCK_SIZE_WITH_PADDING)
NUM_BLOCKS_Y = MAX_BLOCKS_Y - INIT_EMPTY_ROWS


# Custom game class
class Ballz(arcade.Window):
    # Define sprites and game window
    def __init__(self):
        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        # Create main SpriteLists
        self.block_list = arcade.SpriteList(use_spatial_hash=True)
        self.ball_list = arcade.SpriteList()

        # Initial 2D Array representing block values
        self.blockArray = np.zeros((NUM_BLOCKS_Y, NUM_BLOCKS_X), np.int8)

        # Reference to single ball
        self.player = arcade.SpriteCircle(BALL_RADIUS, arcade.color.WHITE)
        self.player.center_x = SCREEN_WIDTH / 2
        self.player.center_y = BALL_INIT_HEIGHT

        # Test emitter
        # Ignore for now
        self.emitter = arcade.Emitter(
            center_xy=(SCREEN_WIDTH/2, BALL_INIT_HEIGHT),
            emit_controller=arcade.EmitterIntervalWithCount(PARTICLE_INTERVAL, PARTICLE_INIT_COUNT),
            particle_factory=lambda emitter: arcade.EternalParticle(
                filename_or_texture="../Resources/ball.png",
                change_xy=(10, 5),
                scale=PARTICLE_SCALE
            )
        )

        arcade.set_background_color(arcade.csscolor.BLACK)

    # Set initial ball angle and create blocks
    def setup(self, angle=40):
        self.setBallAngle(angle)
        self.ball_list.append(self.player)
        self.createBlocksAndArray()

    # Randomly creates blocks
    def createBlocksAndArray(self):
        for x in range(NUM_BLOCKS_X):
            xPos = BLOCK_SIZE_WITH_PADDING * x + BLOCK_OFFSET
            for y in range(NUM_BLOCKS_Y):
                yPos = SCREEN_HEIGHT - BLOCK_OFFSET - BLOCK_SIZE_WITH_PADDING * y

                currVal = self.chooseBlockValue(0.75, 2)
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

    # Moves all blocks on screen down
    def moveBlocksDown(self):
        for block in self.block_list:
            block.center_y -= BLOCK_SIZE_WITH_PADDING

        self.createNewTopRowBlocks()
        self.setAllBlockLabels()
        print(self.blockArray)

    # Creates a new top row of blocks
    def createNewTopRowBlocks(self):
        # End game if there are too many rows of blocks
        if len(self.blockArray) == MAX_BLOCKS_Y:
            arcade.finish_render()
            arcade.close_window()

        newRow = np.zeros([1, NUM_BLOCKS_X], np.int8)

        # Create all block sprites
        for i in range(NUM_BLOCKS_X):
            currVal = self.chooseBlockValue(0.75, 2)
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
        self.blockArray[row][col] -= 1

        val = self.blockArray[row][col]
        # Remove block if the value is 0
        if val == 0:
            block.remove_from_sprite_lists()
        else:
            self.drawBlockLabel(str(val), block.center_x, block.center_y)

    # Simplified code for drawing block value
    # Takes in game coordinates of block, not array coordinates
    def drawBlockLabel(self, val, x, y):
        arcade.draw_text(val, x - 6 * len(val), y - 11, arcade.color.BLACK, FONT_SIZE)

    # Sets the angle of the ball
    # Input is degrees, not radians
    def setBallAngle(self, angle):
        angle = np.deg2rad(angle)
        self.player.change_x = MAX_SPEED*np.cos(angle)
        self.player.change_y = MAX_SPEED*np.sin(angle)

    # Initial render of game
    def on_draw(self):
        arcade.start_render()
        self.block_list.draw()
        self.ball_list.draw()
        self.setAllBlockLabels()
        self.emitter.draw()

    # Physics
    def on_update(self, delta_time: float):
        # self.emitter.update()

        # Moving ball ---------------------------------------------------------------
        self.player.center_x += self.player.change_x * delta_time
        self.player.center_y += self.player.change_y * delta_time

        # Block collision -----------------------------------------------------------
        # Finds all blocks made contact with
        block_hit_list = arcade.check_for_collision_with_list(self.player, self.block_list)
        print(block_hit_list)

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
        elif self.player.center_y <= BALL_INIT_HEIGHT:
            self.player.change_x = 0
            self.player.change_y = 0
            # This marks the end of an "episode"
            # We can add code here to quit the game at this point

            # Ideally at this point, the agent would choose a new angle and fire the ball again.
            # self.moveBlocksDown()
            arcade.pause(1)


def main():
    window = Ballz()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
