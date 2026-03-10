import sys
import os

# Add tools/ directory to path so all module imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))

from PyQt5.QtWidgets import QApplication
from projection_window2 import ProjectionWindow2

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = MainWindow()
    window = ProjectionWindow2()
    window.show()
    sys.exit(app.exec_())
