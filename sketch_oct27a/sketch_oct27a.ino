int trigPin=7; //Sensor Trig pin connected to Arduino pin 7
int echoPin=6;  //Sensor Echo pin connected to Arduino pin 6
float pingTime;  //time for ping to travel from sensor to target and return
float targetDistance; //Distance to Target in inches
float speedOfSound=776.5; //Speed of sound in miles per hour when temp is 77 degrees.
int click = 1;
int led = 13;
int relay = 8;
int alwaysLowAsGND = 9;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(led, OUTPUT);
  pinMode(relay, OUTPUT);
  pinMode(9, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:

  digitalWrite(trigPin, LOW); //Set trigger pin low
  delayMicroseconds(2000); //Let signal settle
  digitalWrite(trigPin, HIGH); //Set trigPin high
  delayMicroseconds(15); //Delay in high state
  digitalWrite(trigPin, LOW); //ping has now been sent
  delayMicroseconds(10); //Delay in low state
  digitalWrite(alwaysLowAsGND, LOW);

  pingTime = pulseIn(echoPin, HIGH);  //pingTime is presented in microceconds
  pingTime=pingTime/1000000; //convert pingTime to seconds by dividing by 1000000 (microseconds in a second)
  pingTime=pingTime/3600; //convert pingtime to hourse by dividing by 3600 (seconds in an hour)
  targetDistance= speedOfSound * pingTime;  //This will be in miles, since speed of sound was miles per hour
  targetDistance=targetDistance/2; //Remember ping travels to target and back from target, so you must divide by 2 for actual target distance.
  targetDistance= targetDistance*63360;    //Convert miles to inches by multipling by 63360 (inches per mile)
  digitalWrite(relay, LOW);

  if(targetDistance < 50 ) {
    digitalWrite(led, HIGH);
      digitalWrite(relay, HIGH);
    Serial.println(click);
    delay(1000); //delay tenth of a  second to slow things down a little.
    digitalWrite(led, LOW);
      digitalWrite(relay, LOW);
    delay(5000);
  }
}

