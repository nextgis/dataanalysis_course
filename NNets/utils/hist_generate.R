trDice = c(2,3,3,4,4,4,5,5,5,5,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,9,9,9,9,10,10,10,11,11,12) 
# auDice = rep(trDice + 4, 3) 
auDice = trDice + 4

breaks = c(2:32)/2

png(filename="DiceProbas1.png")
  hist(trDice, col=rgb(0,0,1,1/4), freq=T, xlim=c(1, 16), ylim=c(0, 7), breaks=breaks, main="Частоты", xlab="Очки")
  hist(auDice, col=rgb(1,0,0, 1/4), freq=T, xlim=c(1, 16), add=T, breaks=breaks) 
dev.off()

