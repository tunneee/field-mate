import { CollectionModel } from "../models/collection.js";

export const createCollection = async (req, res) => {
  try {
    const newCollection = await CollectionModel.create(req.body);
    res.status(201).json(newCollection);
  } catch (error) {
    res.status(400).json({ error: 'Error creating collection.' });
  }
};

export const getCollectionById = async (req, res) => {
  try {
    const collection = await CollectionModel.findById(req.params.id);
    res.status(200).json(collection);
  } catch (error) {
    res.status(404).json({ error: 'Collection not found.' });
  }
};
