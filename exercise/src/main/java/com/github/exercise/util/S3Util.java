package com.github.exercise.util;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.PresignedGetObjectRequest;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;

public class S3Util {
    private final String bucketName;
    private final S3Client s3Client;
    private final S3Presigner s3Presigner;

    public S3Util(final Region region, final String bucketName) {
        this.bucketName = bucketName;
        this.s3Client = S3Client.builder().region(region).build();
        this.s3Presigner = S3Presigner.builder().region(region).build();
    }

    public void putObject(final String key, final byte[] bytes) throws IOException {
        final PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                .bucket(this.bucketName)
                .key(key)
                .build();
        this.s3Client.putObject(putObjectRequest,
                RequestBody.fromInputStream(new ByteArrayInputStream(bytes), bytes.length));
    }

    public void putObject(final String key, final InputStream inputStream, final long contentLength, final String contentType) {
        final PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                .bucket(this.bucketName)
                .key(key)
                .contentType(contentType)
                .build();
        this.s3Client.putObject(putObjectRequest, RequestBody.fromInputStream(inputStream, contentLength));
    }

    public InputStream getObject(final String key) {
        final GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(this.bucketName)
                .key(key)
                .build();
        return this.s3Client.getObject(getObjectRequest);
    }

    public void deleteObject(final String key) {
        final DeleteObjectRequest deleteObjectRequest = DeleteObjectRequest.builder()
                .bucket(this.bucketName)
                .key(key)
                .build();
        this.s3Client.deleteObject(deleteObjectRequest);
    }

    public String generatePresignedUrl(final String key, final int expirationMinutes) {
        final GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(this.bucketName)
                .key(key)
                .build();

        final GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
                .signatureDuration(Duration.ofMinutes(expirationMinutes))
                .getObjectRequest(getObjectRequest)
                .build();

        final PresignedGetObjectRequest presignedRequest = s3Presigner.presignGetObject(presignRequest);
        return presignedRequest.url().toString();
    }

    public void close() {
        if (s3Client != null) {
            s3Client.close();
        }
        if (s3Presigner != null) {
            s3Presigner.close();
        }
    }

}
