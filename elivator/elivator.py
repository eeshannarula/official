import passenger as p;

passenger = p.passenger;

class elivator:
##  constructor function...
    def __init__(self):
##      ever passenger has an id...
        self.Pid = 0;
##      for telling the next turn...
        self.turn = 0;
##      wating list of passengers...
        self.pasangers = [];
##      people in the lift...       
        self.totalPeopleInLift = [];
##      the current floor on which the lift is...   
        self.currentFloor = 1;
##      passenger aloud at a time...        
        self.aloudPasengers = 4;
##      floors ...       
        self.floors = [0,1,2,3,4];
##      total passengers...       
        self.totalPassengers = 0;
## adds a passenger to the wating list...
    def addPassenger(self,onFloor,toFloor):
        self.pasangers.append(passenger(onFloor,toFloor,self.Pid));
        self.Pid+=1;
        self.totalPassengers+=1;
##  number of people in apartcular floor...
    def findProbs(self,floor):
        counter = 0;
        for p in self.pasangers:
            if p.onFloor == floor:
                counter+=1;
        return counter/self.totalPassengers;
##  peopel on a floor in ref. to all floors...
    def probArray(self):
         a = [];
         for f in self.floors:
             p = self.findProbs(f);
             a.append(p);
         return a;
##  this func fined where prob of people want to go...
    def dirsToGoAll(self):
         ups = 0;
         downs = 0;
         for p in self.pasangers:
             if p.toFloor > self.currentFloor:
                 ups+=1;
             else:
                 downs+=1;
         if ups > downs:
             return 0;
         else:
             return 1;

##    functions to drop and pick people forma particular floor...          
##    to pick...
    def pickPeople(self,direction):
        if len(self.totalPeopleInLift) <= 4:
         for p in self.pasangers:
             if p.onFloor == self.currentFloor:
                 d = p.getDir(self.currentFloor);
                 if d == direction:
                    self.totalPeopleInLift.append(self.pasangers.pop(p.no));
##    to drop...
    def  dropPeople(self):
        for i in range(len(self.totalPeopleInLift)):
            p = self.totalPeopleInLift[i];
            if p.toFloor == self.cureentFloor:
                self.totalPeopleInLift.pop(i);

                
                
            

                    

                
                     
                     
                     
                     
                 
                 
                 
        




