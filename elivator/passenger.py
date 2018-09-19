class passenger:
    def __init__(self,onFloor,toFloor,no):
        self.onFloor = onFloor;
        self.toFloor = toFloor;
        self.no = no;

    def getDir(self,current):
        if self > current:
            return 0;
        else:
            return 1;
        
