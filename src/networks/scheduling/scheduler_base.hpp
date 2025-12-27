
class AnnealingScheduler {

	virtual update() = 0;
	virtual get_temp() = 0;
	virtual get_stabilization_its() = 0;

};