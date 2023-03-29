## Displacement of model under uniform load: 1
![Model and it's displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1079524128517804072/image.png)

Material M = STEEL

Material Radius R = 10 m

Material Mass = 1 kg

Notes:

Very little displacement due to high rigidity of the material

## Displacement of model under uniform load: 2
![Model and it's displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1079527739901628446/image.png)

Material M = PVC

Material Radius R = 10 m

Material Mass = 1 kg

Large displacement due to high weight under low rigidity

## Displacement of model under uniform load with springs and base node support + Z support: 3
![Model and it's displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1084830737267695646/image.png)

Material M = STEEL

Material Radius R = varied

Material Mass M = 100 kg

Spring Constant K = 10000 N/m

Very little displacement due to the material

## Displacement of model under uniform load with springs and base node support + Rotational support: 4
![Model and its displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1084840202918432769/image.png)

Material M = STEEL

Material Radius R = varied

Material Mass m = 100kg

Spring Constant K = 30 N/m

Large rotational displacement due to the imbalanced system and low spring constant on base

## Displacement of model under uniform load with springs and base node support + y directional rotation and displacement lock: 5
![Model and its displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1085211678368596018/image.png)

Material M = POLYETHYLENE

Material Mass m = 100kg

Spring Constant K = 10000 N/m

Large displacement due to gravitational force on such a weak material

## Displacement of model under uniform load with springs and base node support + y directional rotation and displacement lock: 6
![Model and it's displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1085222400221786194/image.png)

Material M = STEEL

Material Mass m = 100kg

Spring Constant K = 10000 N/m

Sagging and some displacement due to huge mass on weak spring (Previous iterations did not properly utilise spring supports)

### Same as previous but with K = 100000

![Model and it's displacement under uniform load](https://cdn.discordapp.com/attachments/553961513686269975/1085225723310129192/image.png)


# Final Stages Version Testing

## Model of light concrete L beam without support

![Model and it's displacement under uniform gravitational load](https://cdn.discordapp.com/attachments/553961513686269975/1090647490430246952/image.png)

Material M = Light Concrete

Material Mass m = 10kg

Spring Constant K = 1e30 N/m

Deforms considerably without support

## Model of light concrete L beam with steel support

![Model and it's displacement under uniform gravitational load](https://cdn.discordapp.com/attachments/553961513686269975/1090651266289500241/image.png)

Material M = Light Concrete

Support Material M2 = Steel (Running through the internal section of the beam)

Material Mass m = 10kg

Spring Constant K = 1e30 N/m

Deforms still with support but not as considerably

## Model of steel L beam for reference

![Model and it's displacement under uniform gravitational load](https://cdn.discordapp.com/attachments/553961513686269975/1090648027565412412/image.png)

Material Mass m = 10kg

Spring Constant K = 1e30 N/m

Deforms much less than concrete with or without support
