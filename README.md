# Frog ID (prototype)

Identify East Texas frogs and toads by their calls, in the browser:
**https://lukbio.github.io/frog-id-web/**

- **Listen**: hold your phone up to a chorus. Every 3 seconds it names every
  frog that passes the confidence setting (or says "not sure" / "no frog"),
  with a live spectrogram and a list of what it heard this session.
- **Analyze a recording**: pick an audio file (m4a, mp3, wav) and get an answer
  for every 3 seconds plus an overall top 3.
- **Near me**: uses your location (rounded, kept on your phone) and today's
  date to rule out frogs with no sightings nearby or that don't call this month.
- **Species**: the 24 East Texas frogs and toads (23 in the model; the Houston Toad has too few recordings), with calling seasons and
  links to iNaturalist.

Everything runs on the device. Audio is never uploaded. After the first visit
it works with no signal, and Safari's Share → Add to Home Screen turns it into
an app icon.

## How it works

Each 3-second window (48 kHz mono) goes through BirdNET V2.4, cut down to its
1,024-number embedding (`model/birdnet_embed.tflite`, run with TensorFlow Lite
for the web). A logistic-regression classifier (`model/frog_classifier.json`)
trained on about 5,000 iNaturalist frog recordings, plus insects, night birds,
rain, wind and traffic, gives each species its own score. `model/filter.json` holds
the range grid (half-degree cells) and calling months.

In cross-validated testing it named the right species 74% of the time on the
first guess, and was right 90% of the time when at least 70% confident (47% of
recordings). It also trains on made-up mixes of two frogs, so it can name more
than one; on test mixes it names both 16% of the time, so choruses are still
its weak spot.

## Credits and licences

- BirdNET V2.4 model by the K. Lisa Yang Center for Conservation Bioacoustics
  (Cornell Lab of Ornithology) and Chemnitz University of Technology,
  CC BY-NC-SA 4.0 (`model/BIRDNET_LICENSE.txt`).
- Training and sample recordings: iNaturalist contributors (credited in the
  app for each sample) and ESC-50 (K. J. Piczak, 2015), CC BY-NC.
- Range and season data: iNaturalist research-grade observations.
- TensorFlow.js and TFLite for the web (`vendor/`): Google, Apache 2.0.

Non-commercial research and education use only.
