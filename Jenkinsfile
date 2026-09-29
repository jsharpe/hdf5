node('c++') {
  stage 'Build and Test'
  checkout scm
  dir('build') {
    withEnv(['PATH=/opt/kitware/cmake-3.4.1-Linux-x86_64/bin:/opt/rh/devtoolset-3/root/usr/bin/:$PATH']) {
        sh '''cmake ..
              make
              make test'''
    }
  }
}