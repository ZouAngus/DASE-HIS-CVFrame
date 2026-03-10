import sys
from PyQt5.QtWidgets import QApplication
from projection_window2 import ProjectionWindow2

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = MainWindow()
    window = ProjectionWindow2()
    window.show()
    sys.exit(app.exec_())
