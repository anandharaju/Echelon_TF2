  library("devtools")
  library("nonpar")
  
  cutoff = 0.015
  
  vec <- c('ndata', 'shared', 'debug', 'tls', 'itext', 'cdata', 'oroc', 'stab', 'stabstr', 'edata', 'srdata', 'sdata', 'npdata', 'textbss', 'detourd', 'didata', 'imrsiv', 'shdata', 'isoapis')
  vec <- c('header', 'text', 'data', 'pdata', 'rsrc', 'rdata', 'edata', 'idata', 'bss', 'reloc', 'debug', 'sdata', 'hdata', 'xdata', 'npdata', 'itext', 'apiset', 'qtmetad', 'textbss', 'its', 'extjmp', 'cdata', 'detourd', 'cfguard', 'guids', 'sdbid', 'extrel', 'ndata', 'detourc', 'shared', 'rodata', 'gfids', 'didata', 'pr0', 'tls', 'imrsiv', 'stab', 'mrdata', 'sxdata', 'orpc', 'c2r', 'nep', 'shdata', 'srdata', 'didat', 'stabstr', 'bldvar', 'isoapis')
  keys <- array(vec, dim=c(1,length(vec)))
  len = length(keys)

  for(i in 1:len){
    b <- read.csv(paste("D:\\03_GitWorks\\echelon\\out\\result\\benign.csv", ".activation_", keys[i], ".csv", sep = ""), stringsAsFactors = FALSE, header = FALSE, sep=",")
    m <- read.csv(paste("D:\\03_GitWorks\\echelon\\out\\result\\malware.csv", ".activation_", keys[i], ".csv", sep = ""), stringsAsFactors = FALSE, header = FALSE, sep=",")
    b <- as.double(b[1,])
    m <- as.double(m[1,])
    
    b <- b[b>cutoff]
    m <- m[m>cutoff]
    
    print(cat(length(b) , length(m) , keys[i]))

    b <- (b - min(b))/(max(b) - min(b))
    m <- (m - min(m))/(max(m) - min(m))


    
    print(cat(length(b) , length(m) , keys[i]))
    cuc = cucconi.test(x = b, y = m, method="permutation")
    print(cuc)
    #break
  }
  
  
  
  
  
  
  
  
  

