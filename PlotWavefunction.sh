#!/bin/bash

for input in "${@:2}"; do
    cp $1 tmp.xcrysden
    filename="${input%.*}.png"
    echo '

# close Isosurface control window
# set windowID [exec xwininfo -tree -root | grep "Isosurface/Property-plane Controls" | sed "s/^ *//g" | cut -d " " -f 1]
set allWindows [exec xwininfo -tree -root]
set isoWindow [exec echo $allWindows | grep "Isosurface/Property-plane Controls" | sed "s/^ *//g"]
set mainWindow [exec echo $allWindows | grep "XCrySDen" | sed "s/^ *//g"]
set isoWindowID [exec echo $isoWindow | cut -d " " -f 1]

set mainWindowSize [exec echo $mainWindow | grep "\[0-9\]*x\[0-9\]*+\[0-9\]*+\[0-9\]*" -o | cut -d "+" -f 1]
set xMove [exec echo $mainWindowSize | cut -d "x" -f 1]
set yMove [exec echo $mainWindowSize | cut -d "x" -f 2]

exec xwit -id $isoWindowID -move $xMove $yMove

# Print
scripting::printToFile '"$filename
exit 0" >> tmp.xcrysden;
    xcrysden --cube $input --script tmp.xcrysden || exit 1;
done
rm -f tmp.xcrysden;
