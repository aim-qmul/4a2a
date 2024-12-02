/*
  ==============================================================================

	This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "PluginProcessor.h"
#include <JuceHeader.h>
#include <algorithm>
#include <array>

typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;

//==============================================================================
/**
 */
class _4A2AAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
	_4A2AAudioProcessorEditor(_4A2AAudioProcessor &, juce::AudioProcessorValueTreeState &);
	~_4A2AAudioProcessorEditor() override;

	//==============================================================================
	void paint(juce::Graphics &) override;
	void resized() override;

private:
	// This reference is provided as a quick way for your editor to
	// access the processor object that created it.
	_4A2AAudioProcessor &audioProcessor;
	juce::AudioProcessorValueTreeState &valueTreeState;

	std::unique_ptr<SliderAttachment> thresholdAttachment, ratioAttachment,
		attackMsAttachment, releaseMsAttachment, makeUpAttachment;
	juce::Slider peakReduction, threshold, ratio, attackMs, releaseMs, makeUp;
	juce::Label peakReductionLabel, thresholdLabel, ratioLabel, attackMsLabel,
		releaseMsLabel, makeUpLabel;

	float peakPoints[13] = {40, 45, 50, 55, 60, 65, 70,
							75, 80, 85, 90, 95, 100};
	float paramPoints[13][5] = {
		{-13.075952529907227, 5.435610294342041, 30.516525268554688,
		 470.176513671875, 0.697806715965271},
		{-14.217720985412598, 4.510651588439941, 17.315229415893555,
		 371.4613342285156, 0.6503340005874634},
		{-15.320412635803223, 4.193428039550781, 11.579639434814453,
		 324.1263427734375, 0.6253266930580139},
		{-16.131834030151367, 4.151314735412598, 8.878186225891113,
		 300.0677795410156, 0.6053885817527771},
		{-16.88168716430664, 4.212928295135498, 7.19603967666626,
		 283.0182189941406, 0.5758121609687805},
		{-17.586275100708008, 4.265491962432861, 5.948177814483643,
		 266.1832580566406, 0.5347366333007812},
		{-19.835796356201172, 4.289010047912598, 3.8497214317321777,
		 254.91220092773438, 0.4238012433052063},
		{-21.904220581054688, 4.237095355987549, 2.8647751808166504,
		 242.99441528320312, 0.33676016330718994},
		{-24.04867935180664, 4.137104511260986, 2.2430710792541504,
		 234.94378662109375, 0.29632773995399475},
		{-27.549327850341797, 3.9736087322235107, 1.7689090967178345,
		 248.10983276367188, 0.047231659293174744},
		{-31.510976791381836, 3.874948501586914, 1.265961766242981,
		 245.48947143554688, 0.03817363455891609},
		{-33.69828796386719, 3.791938066482544, 1.0862258672714233,
		 244.5567626953125, 0.1468079835176468},
		{-35.34008026123047, 3.711195945739746, 0.9788835048675537,
		 244.128662109375, 0.3002852201461792}};

	std::array<float, 5> interp(float peakValue);

	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(_4A2AAudioProcessorEditor)
};
