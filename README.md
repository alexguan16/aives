<div align="center">

<img src="https://raw.githubusercontent.com/alexguan16/aives/ml/aves_logo.svg" alt='Aves logo' width="200" />

## Aives

Aives is a fork of Aves, with support for contextual image searching, or AI powered image searching
      
<div align="left">

## Features

Aives processes images into embeds during gallery analysis and cataloging

Aives generates text embeds off user queries which are compared against the image embeds to return images sorted by similarity

Aives is a fork of Aves, whos main features can be found here: [AVES MAIN PROJECT REPO](https://github.com/deckerst/aves)

## Model

Aives currently bundles a non quantized MobileClip2-s0 for its on device inference

MobileClip2-s0 was mainly chosen for its size and accuracy tradeoff allowing for a smaller, but still substantial 460mb apk

## Screenshots

<div align="center">

<img height="600" alt="image" src="https://github.com/user-attachments/assets/a4bf578e-6413-47ec-8902-5afcbb7d99d8" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d67f53ac-ef79-4ba8-9365-c80e650893bf" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/7ac4ff5e-844c-4e11-9bfc-de0e376d7355" />



<div align="left">

## Permissions

Aives requires a few permissions to do its job:
- **ignore battery restrictions**: allows the app to continue processing embeds in the background,
- **read contents of shared storage**: the app only accesses media files, and modifying them requires explicit access grants from the user,
- **read locations from media collection**: necessary to display the media coordinates, and to group them by country (via reverse geocoding),

## Project Setup

Before running or building the app, update the dependencies for the libre flavor:
```
# scripts/apply_flavor_libre.sh
```

To build the project, create a file named `<app dir>/android/key.properties`. It should contain a reference to a keystore for app signing, and other necessary credentials. See [key_template.properties](https://github.com/deckerst/aves/blob/develop/android/key_template.properties) for the expected keys.

To run the app:
```
# ./flutterw run -t lib/main_libre.dart --flavor libre
```
