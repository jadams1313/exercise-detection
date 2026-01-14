package com.github.repositories;

import com.github.data.Excercise;
import org.springframework.data.domain.Example;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.FluentQuery;

import java.util.List;
import java.util.Optional;
import java.util.function.Function;

public class ExcerciseRepo implements JpaRepository<Excercise, Long> {
    @Override
    public void flush() {

    }

    @Override
    public <S extends Excercise> List<S> saveAllAndFlush(Iterable<S> entities) {
        return List.of();
    }

    @Override
    public void deleteAllInBatch(Iterable<Excercise> entities) {

    }

    @Override
    public void deleteAllByIdInBatch(Iterable<Long> longs) {

    }

    @Override
    public void deleteAllInBatch() {

    }

    @Override
    public Excercise getOne(Long aLong) {
        return null;
    }

    @Override
    public Excercise getById(Long aLong) {
        return null;
    }

    @Override
    public Excercise getReferenceById(Long aLong) {
        return null;
    }

    @Override
    public <S extends Excercise> Optional<S> findOne(Example<S> example) {
        return Optional.empty();
    }

    @Override
    public <S extends Excercise> List<S> findAll(Example<S> example) {
        return List.of();
    }

    @Override
    public <S extends Excercise> List<S> findAll(Example<S> example, Sort sort) {
        return List.of();
    }

    @Override
    public <S extends Excercise> Page<S> findAll(Example<S> example, Pageable pageable) {
        return null;
    }

    @Override
    public <S extends Excercise> long count(Example<S> example) {
        return 0;
    }

    @Override
    public <S extends Excercise> boolean exists(Example<S> example) {
        return false;
    }

    @Override
    public <S extends Excercise, R> R findBy(Example<S> example, Function<FluentQuery.FetchableFluentQuery<S>, R> queryFunction) {
        return null;
    }

    @Override
    public <S extends Excercise> S saveAndFlush(S entity) {
        return null;
    }

    @Override
    public <S extends Excercise> S save(S entity) {
        return null;
    }

    @Override
    public <S extends Excercise> List<S> saveAll(Iterable<S> entities) {
        return List.of();
    }

    @Override
    public Optional<Excercise> findById(Long aLong) {
        return Optional.empty();
    }

    @Override
    public boolean existsById(Long aLong) {
        return false;
    }

    @Override
    public List<Excercise> findAll() {
        return List.of();
    }

    @Override
    public List<Excercise> findAllById(Iterable<Long> longs) {
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
    public void delete(Excercise entity) {

    }

    @Override
    public void deleteAllById(Iterable<? extends Long> longs) {

    }

    @Override
    public void deleteAll(Iterable<? extends Excercise> entities) {

    }

    @Override
    public void deleteAll() {

    }

    @Override
    public List<Excercise> findAll(Sort sort) {
        return List.of();
    }

    @Override
    public Page<Excercise> findAll(Pageable pageable) {
        return null;
    }
}
