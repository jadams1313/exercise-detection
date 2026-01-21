package com.github.exercise.repositories;

import com.github.exercise.constants.VideoStatus;
import com.github.exercise.data.User;
import com.github.exercise.data.VideoUpload;
import org.springframework.data.domain.Example;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.FluentQuery;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

public class VideoUploadRepo implements JpaRepository<VideoUpload, Long> {
    @Override
    public void flush() {

    }

    @Override
    public <S extends VideoUpload> S saveAndFlush(S entity) {
        return null;
    }

    @Override
    public <S extends VideoUpload> List<S> saveAllAndFlush(Iterable<S> entities) {
        return List.of();
    }

    @Override
    public void deleteAllInBatch(Iterable<VideoUpload> entities) {

    }

    @Override
    public void deleteAllByIdInBatch(Iterable<Long> longs) {

    }

    @Override
    public void deleteAllInBatch() {

    }

    @Override
    public VideoUpload getOne(Long aLong) {
        return null;
    }

    @Override
    public VideoUpload getById(Long aLong) {
        return null;
    }

    @Override
    public VideoUpload getReferenceById(Long aLong) {
        return null;
    }

    @Override
    public <S extends VideoUpload> Optional<S> findOne(Example<S> example) {
        return Optional.empty();
    }

    @Override
    public <S extends VideoUpload> List<S> findAll(Example<S> example) {
        return List.of();
    }

    @Override
    public <S extends VideoUpload> List<S> findAll(Example<S> example, Sort sort) {
        return List.of();
    }

    @Override
    public <S extends VideoUpload> Page<S> findAll(Example<S> example, Pageable pageable) {
        return null;
    }

    @Override
    public <S extends VideoUpload> long count(Example<S> example) {
        return 0;
    }

    @Override
    public <S extends VideoUpload> boolean exists(Example<S> example) {
        return false;
    }

    @Override
    public <S extends VideoUpload, R> R findBy(Example<S> example, Function<FluentQuery.FetchableFluentQuery<S>, R> queryFunction) {
        return null;
    }

    @Override
    public <S extends VideoUpload> S save(S entity) {
        return null;
    }

    @Override
    public <S extends VideoUpload> List<S> saveAll(Iterable<S> entities) {
        return List.of();
    }

    @Override
    public Optional<VideoUpload> findById(Long aLong) {
        return Optional.empty();
    }

    @Override
    public boolean existsById(Long aLong) {
        return false;
    }

    @Override
    public List<VideoUpload> findAll() {
        return List.of();
    }

    @Override
    public List<VideoUpload> findAllById(Iterable<Long> longs) {
        return List.of();
    }

    @Override
    public long count() {
        return 0;
    }

    @Override
    public void deleteById(Long aLong) {

    }

    @Override
    public void delete(VideoUpload entity) {

    }

    @Override
    public void deleteAllById(Iterable<? extends Long> longs) {

    }

    @Override
    public void deleteAll(Iterable<? extends VideoUpload> entities) {

    }

    @Override
    public void deleteAll() {

    }

    @Override
    public List<VideoUpload> findAll(Sort sort) {
        return List.of();
    }

    @Override
    public Page<VideoUpload> findAll(Pageable pageable) {
        return null;
    }

    public List<VideoUpload> findByUserOrderByUploadedAtDesc(User user) {
        return Collections.EMPTY_LIST;
    }

    public List<VideoUpload> findByStatus(VideoStatus videoStatus) {
        return null;
    }
}
