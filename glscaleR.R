
require(optimx)
# General Logistic Function
glf <- function(x, const) {
    B = const[1]
    M = const[2]
    Q = const[3]
    v = const[4]
    out = 1 / (1 + Q * exp(-B * (x - M))) ^ (1 / v)
    return(out)
}
# Objective function
obj.f <- function(x, const) {
    fval = 0
    Fn = ecdf(x)
    for (i in x) {
        fval = fval + (Fn(i) - glf(i, const)) ^ 2
    }
    return(fval)
    
}

# Gradient of the objective function
obj.g = function(x, const) {
    
    B = const[1]
    M = const[2]
    Q = const[3]
    v = const[4]
    
    dB = 0
    dM = 0
    dQ = 0
    dv = 0
    
    Fn = ecdf(x)
    
    for (xi in x) {
        
        T0 = exp(- B * (xi - M))
        T1 = 2 * (Fn(xi) - glf(xi,const))
        T2 = v * (Q * T0 + 1) ^ (1 / v + 1)
        
        dB = dB - T1 * Q * T0 * (xi - M) / T2
        dM = dM + T1 * B * Q * T0 / T2
        dQ = dQ + T1 * T0 / T2
        dv = dv - T1 * log(Q * T0 + 1) / (v ^ 2 * (Q * T0 + 1) ^ (1 / v))
        g = c(dB, dM, dQ, dv)
        return(g)
        
    }
    
}

glscale <- function(data, setting = NULL) {
    # data is a matrix+
    temp = dim(data)
    nr = temp[1]
    nc = temp[2]
    
    out = matrix(, nrow = nr, ncol = nc)
    
    if (is.null(setting)) {
        
        setting = matrix(, nrow = 4, ncol = nc)
        for (i in 1:nc) {
            
            x = data[, i]
            
            xmin = min(x)
            xmax = max(x)
            xmed = median(x)
            
            M0 = xmed
            Q0 = 1
            B0 = (log((1 + Q0) ^ 3.3219 - 1) - log(Q0)) / (M0 - xmin)
            v0 = log(1 + Q0) / log(2)
            
            init = c(B0, M0, Q0, v0)
            f = function(const) {
                return (obj.f(x, const))
            }
            g = function(const) {
                return (obj.g(x, const))
            }
            # opt = optimx(init, f, g, method = "BFGS")
            opt = optimx(init, f, method = "BFGS")
            setting[,i] = c(opt$p1,opt$p2,opt$p3,opt$p4)
            out[, i] = glf(x, setting[,i])
            
        }
        
    }
    
    else {
        for (i in 1:nc) {
            x = data[, i]
            out[, i] = glf(x, setting[,i])
            
        }
        
    }
    return(list(out, setting))
    
}

## Toy example

x = matrix( rnorm(100*3,mean=0,sd=1), 100, 3) 

out = glscale(x)
newx = out[[1]]
setting = out[[2]]

testx = matrix( rnorm(20*3,mean=0,sd=1), 20, 3) 
testOut = glscale(testx,setting)