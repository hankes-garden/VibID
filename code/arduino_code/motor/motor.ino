#include <stdio.h>

const int ANALOG_OUT_PIN = 3;
const int INTENSITY_STEP = 5;
const int VIBRATION_DURATION = 1000;
const int INTERVAL_DURATION = 100;
const int RESET_DURATION = 5000;
const int SERIAL_BAUD_RATE = 9600;
const int MAX_INTENSITY = 200;

bool g_bSweep = false;
int g_nCurrentIntensity = 0;
int g_nMaxIntensity = 255;
char g_arrStrFormatBuffer [50] = {'\0'};

void setup() {
  Serial.begin(SERIAL_BAUD_RATE); // opens serial port, sets data rate to 9600 bps
}

void loop() {

  if (Serial.available() > 0) 
  {
    int nInput = Serial.parseInt();

    if (nInput == 0) // 0: stop to vibrate
    {
      Serial.println("-->stop vibration.");
      g_bSweep = false;
      g_nCurrentIntensity = 0;
      setVibration(0);
    }
    else if (nInput > 0) // vibrate as user input for one duration
    {
      Serial.println("-->start to vibrate...");

      // stop vibration first
      g_bSweep = false;
      setVibration(0);
      delay(INTERVAL_DURATION);

      // vibrate
      setVibration(nInput);
      delay(VIBRATION_DURATION);
      setVibration(0);
    }
    else // < 0, sweep from 0 to specific input
    {
      g_nMaxIntensity = min(abs(nInput), MAX_INTENSITY); // set maximum of sweeping
      g_nCurrentIntensity = 0;
      g_bSweep = true;
      Serial.println("-->start to sweep...");
      delay(RESET_DURATION);
    }
  }

  if (g_bSweep == true)// change vibration periodically
  {
    if (g_nCurrentIntensity <= g_nMaxIntensity)
    {
      // vibrate
      setVibration(g_nCurrentIntensity);
      delay(VIBRATION_DURATION);

//      // stop for a while
//      setVibration(0);
//      delay(INTERVAL_DURATION);

      g_nCurrentIntensity += INTENSITY_STEP;
    }
    else
    {
      Serial.println("-->sweep is over");
      g_nCurrentIntensity = 0;
      setVibration(g_nCurrentIntensity);
      g_bSweep = false;
      delay(RESET_DURATION);
    }
  }

}

/*
 * This function set the vibration intensity via PWM
 */
void setVibration(int nIntensity)
{
  int nVol = min(nIntensity, MAX_INTENSITY);
  analogWrite(ANALOG_OUT_PIN,  nVol);

  sprintf(g_arrStrFormatBuffer, "current intensity: %d.", nVol);
  Serial.println(g_arrStrFormatBuffer);
}




