import arcade
import numpy as np

# Constants
SCREEN_WIDTH = 360
SCREEN_HEIGHT = 480
SCREEN_TITLE = "Ballz v1"
BLOCK_SCALING = 0.6

# Initital 2D Array representing blocks
blockArray = [[0, 0, 0, 0, 0, 10, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 2, 2, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 4, 0, 0],
              [0, 0, 2, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]

class MyGame(arcade.Window):
    def __init__(self):
        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        # Create main SpriteLists
        self.block_list = arcade.SpriteList(use_spatial_hash=True)
        self.ball_list = arcade.SpriteList()

        # Reference to single ball
        self.player = None

        arcade.set_background_color(arcade.csscolor.BLACK)

    def setup(self):
        #Create single ball sprite
        self.player = arcade.SpriteCircle(5, arcade.color.WHITE)
        self.player.center_x = SCREEN_WIDTH/2
        self.player.center_y = 30

        #Set initial speed
        self.player.change_x = 300
        self.player.change_y = 120
        self.ball_list.append(self.player)

        self.createBlocksFromArray()

        #self.physics_engine = arcade.PhysicsEngineSimple(self.player, self.block_list)

    #Creates blocks in position using blockArray
    def createBlocksFromArray(self):
        for x in range(9):
            xPos = 40 * x + 19 # Fit 9 columns
            for y in range(10):
                yPos = 40 * y + 100 # Fit 10 rows

                # (0,0) is top left in array, but bottom-left in window
                # Without the subtraction, the blocks in the display would appear
                # to be "vertically flipped"
                currVal = blockArray[len(blockArray) - y][x]

                if currVal != 0:
                    # Create block object
                    block = arcade.SpriteSolidColor(36, 36, (0, 255, 0))
                    block.center_x = xPos
                    block.center_y = yPos
                    self.block_list.append(block)

    # Find the corresponding index in the array for a box on display
    def getBlockIndexInArray(self, block):
        x = int(np.round((block.center_x - 19)/40))
        y = int(np.round((block.center_y - 100)/40))

        return (9 - y), x

    # Overlays text labels on top of blocks in the display
    def updateBallLabels(self):
        for x in range(9):
            xPos = 40 * x + 19
            for y in range(10):
                yPos = 40 * y + 100

                val = str(blockArray[9 - y][x])

                if val != "0":
                    arcade.draw_text(val, xPos-6*len(val), yPos-11, arcade.color.BLACK, 18)

    # Initial render of game
    def on_draw(self):
        arcade.start_render()
        self.block_list.draw()
        self.ball_list.draw()
        self.updateBallLabels()

    # Physics
    def on_update(self, delta_time: float):
        #self.physics_engine.update()
        
        # Ball Physics
        self.player.center_x += self.player.change_x * delta_time
        self.player.center_y += self.player.change_y * delta_time
        
        # Wall Bounce
        if self.player.center_x < 5 or self.player.center_x > SCREEN_WIDTH - 5:
            self.player.change_x *= -1
        if self.player.center_y < 5 or self.player.center_y > SCREEN_HEIGHT - 5:
            self.player.change_y *= -1

        # Block physics
        block_hit_list = arcade.check_for_collision_with_list(self.player, self.block_list)

        for block in block_hit_list:
            collision_list = (abs(block.top - self.player.center_y),
                              abs(block.bottom - self.player.center_y),
                              abs(block.right - self.player.center_x),
                              abs(block.left - self.player.center_x))

            index = np.argmax(collision_list)

            if index == 0 or index == 1:
                self.player.change_y *= -1
            elif index == 2 or index == 3:
                self.player.change_x *= -1

            # Reduce block's value by one
            row, col = self.getBlockIndexInArray(block)
            blockArray[row][col] -= 1
            
            # Remove if block has a value of 0
            if blockArray[row][col] == 0:
                block.remove_from_sprite_lists()

            self.updateBallLabels()

def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
