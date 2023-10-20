import { DiseaseModel } from "../models/disease.js";

export const createDisease = async (req, res) => {
  try {
    const newDisease = await DiseaseModel.create(req.body);
    res.status(201).json(newDisease);
  } catch (error) {
    res.status(400).json({ error: 'Error creating disease.' });
  }
};

export const getDiseaseById = async (req, res) => {
  try {
    const disease = await DiseaseModel.findById(req.params.id);
    res.status(200).json(disease);
  } catch (error) {
    res.status(404).json({ error: 'Disease not found.' });
  }
};
