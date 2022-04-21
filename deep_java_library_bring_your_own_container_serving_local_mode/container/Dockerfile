FROM openjdk:11

COPY src/ src/
COPY gradle/ gradle/
COPY build.gradle settings.gradle gradlew ./

RUN ./gradlew assemble

ENTRYPOINT ["java","-jar","./build/libs/sagemaker-djl-serving-0.0.1-SNAPSHOT.jar"]